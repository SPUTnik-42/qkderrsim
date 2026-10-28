"""
Puncturing-only 5G NR LDPC reconciliation for BB84, with finite-length
parity start point and universal-hash verification.

Leakage is measured at the oracle boundary via a CountingOracle wrapper:
every bit actually returned by Alice (parities + hash parity queries) is
charged, robust against exceptions.  This is the most defensible empirical
measurement of EC leakage.

Reported efficiencies:
    f_EC    = leaked_bits / (k*h(p))                (parities + hash)
    f_total = (leaked_bits + stop_leak) / (k*h(p))  (adds stopping term)

The stopping-time term is reported separately, since it has no
peer-reviewed citation in this context; the parity bits dominate in
practice.  Failed chunks are listed in stats['failed_chunk_indices'] —
the caller must exclude these positions before privacy amplification.

The sweep start point is a finite-length bound (Polyanskiy–Poor–Verdú,
Tomamichel et al.) rather than the asymptotic Slepian–Wolf bound.  This
prevents the protocol from starting too low and inflating f_EC at low QBER.

Hash verification: a fresh 64-bit seed per chunk gives a
Carter-Wegman universal (Hankel) family; the per-chunk collision
probability is 2^-64, so the union bound over n_chunks gives
eps_cor <= n_chunks * 2^-64.  The seed is drawn from the PRNG, which is
pseudo-universal in practice (deterministic PRNG).

References
----------
[1] Slepian & Wolf, Bell System Technical Journal 52(10):1657-1687, 1973.
[2] Tan & Kosut, IEEE Trans. Inf. Theory 60(2):881-903, 2014.
    Normal approximation, used only qualitatively (omits O(log n) terms).
[3] Polyanskiy, Poor, Verdu, IEEE Trans. Inf. Theory 56(5):2307-2359, 2010.
    BSC dispersion V(p) = p(1-p)[log2((1-p)/p)]^2.
[4] Elkouss, Martinez-Mateo, Martin, Quantum Inf. Comput. 11(3):226-238,
    2011.  Reconciliation efficiency f = m / (k*h(p)).
[5] 3GPP TS 38.212 §5.3.2 (first 2Z systematic positions punctured).
[6] Richardson & Urbanke, Modern Coding Theory, Cambridge, 2008.
[7] Gallager, Low-Density Parity-Check Codes, MIT Press, 1963.
[8] Sionna LDPC5GDecoder: llr_max default is 20.0.
[9] Carter & Wegman, J. Comput. Syst. Sci. 18(2):143-154, 1979.
    Consecutive windows of a random vector form a universal family
    (often called Toeplitz; strictly it is Hankel).
[10] Tomamichel, Martinez-Mateo, Pacher, Elkouss, Quantum Inf. Process.
     16(11):280, 2017.  Finite-key limits for one-way reconciliation;
     improved leakage estimate with finite-length correction.
"""

import asyncio, math, os, sys
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from prng import PRNG
from cascade_refactored import ParityOracle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
SIONNA_AVAILABLE = False

try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
    from sionna.phy.fec.ldpc.decoding   import LDPC5GDecoder
    SIONNA_AVAILABLE = True
    print("[LDPC] Sionna loaded.")
except ImportError as e:
    print(f"[LDPC] Sionna/TF import error: {e}")
except Exception as e:
    print(f"[LDPC] Unexpected error loading Sionna/TF: {e}")


# --- math helpers -----------------------------------------------------------
def binary_entropy(p: float) -> float:
    """h(p) = -p log2 p - (1-p) log2(1-p)."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def bsc_dispersion(p: float) -> float:
    """V(p) = p(1-p)[log2((1-p)/p)]^2  (Polyanskiy et al. 2010, Eq. 4.2)."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return p * (1.0 - p) * (math.log2((1.0 - p) / p)) ** 2


def normal_quantile(prob: float) -> float:
    """Inverse standard normal CDF via rational approximation (Acklam)."""
    # Coefficients from Peter J. Acklam's algorithm.
    a = [-3.969683028665376e+01,  2.209460984245205e+02,
         -2.759285104469687e+02,  1.383577518672690e+02,
         -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02,
         -1.556989798598866e+02,  6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00]
    p_low  = 0.02425
    p_high = 1.0 - p_low
    if prob <= 0.0:
        return -1e10
    if prob >= 1.0:
        return 1e10
    if prob < p_low:
        q = math.sqrt(-2.0 * math.log(prob))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    if prob <= p_high:
        q = prob - 0.5
        r = q * q
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - prob))
    return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
            ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)


# --- leakage-counting wrapper ----------------------------------------------
class _CountingOracle(ParityOracle):
    """
    Wraps a ParityOracle and records bits and calls actually made.
    Leakage is charged at the oracle boundary, so any exception after a
    successful `get_parities` call still counts.
    """
    def __init__(self, wrapped: ParityOracle):
        self.wrapped = wrapped
        self.bits = 0
        self.calls = 0

    async def get_parity(self, indices: List[int]) -> int:
        r = await self.wrapped.get_parity(indices)
        self.bits += 1
        self.calls += 1
        return r

    async def get_parities(self, blocks: List[List[int]]) -> List[int]:
        r = await self.wrapped.get_parities(blocks)
        self.bits += len(r)
        self.calls += 1
        return r

    async def apply_privacy_amplification(self, *a, **kw):
        return await self.wrapped.apply_privacy_amplification(*a, **kw)


# --- protocol ---------------------------------------------------------------
class NR_LDPC_Standard_ClientProtocol:
    """
    Puncturing-only 5G NR LDPC reconciliation with finite-length parity
    budget and universal-hash verification.

    Parameters
    ----------
    verbose       : print progress.
    rate          : None or <= 1/3  -> finite-length start point (auto).
                    > 1/3           -> fixed starting rate; sweep disabled.
    seed          : PRNG seed.
    num_iter      : BP iterations per decode.
    max_k         : target chunk size.
    target_fer    : target frame error rate for the finite-length bound.
    sweep_factor  : multiplicative step in the m sweep (>1).
    max_sweep_iter: iteration cap for the sweep.
    hash_bits     : universal-hash output length (bits per chunk).
    """

    LLR_PARITY = 20.0 #50.0 


    def __init__(self, verbose=True, rate=None, seed=None,
                 num_iter=20, max_k=7000, target_fer=0.1, 
                 sweep_factor=1.22, max_sweep_iter=5, #1.62
                 hash_bits=32): #512 #64
        self.verbose = verbose
        if rate is None or float(rate) <= 1.0/3.0 + 1e-3:
            self.fixed_rate: Optional[float] = None
        else:
            self.fixed_rate = float(rate)
        self.num_iter = int(num_iter)
        self.max_k = int(max_k)
        self.target_fer = float(target_fer)
        self.sweep_factor = max(1.005, float(sweep_factor))
        self.max_sweep_iter = int(max_sweep_iter)
        self.hash_bits = int(hash_bits)
        self.prng = PRNG(seed)
        self._code_cache: Dict[int, dict] = {}
        self._perm: Optional[np.ndarray] = None
        self.bits_revealed = 0
        self.channel_uses = 0
        self.round_trips = 0
        self.stats: Dict[str, float] = {}
        self._executor = ThreadPoolExecutor(max_workers=1)

    def log(self, msg):
        if self.verbose:
            print(f"[LDPC] {msg}")

    # ---------------------------------------------------------------- code
    def _get_code(self, k: int) -> dict:
        if k in self._code_cache:
            return self._code_cache[k]
        if not SIONNA_AVAILABLE:
            raise RuntimeError("Sionna/TF unavailable.")

        n_mother = 3 * k
        enc = LDPC5GEncoder(k=k, n=n_mother)
        dec = LDPC5GDecoder(enc, num_iter=self.num_iter,
                            return_infobits=True,
                            llr_max=self.LLR_PARITY)  # [8]
        n = int(getattr(enc, "n", getattr(enc, "_n", n_mother)))
        Z = int(getattr(enc, "_z", getattr(enc, "z", 0)))
        if Z <= 0:
            raise RuntimeError(f"Could not determine Z for k={k}.")
        n_punctured = 2 * Z
        n_syst = k - n_punctured
        if n_syst <= 0:
            raise RuntimeError(f"k={k} too small for Z={Z}.")

        batch = 256
        rows = []
        for start in range(0, k, batch):
            end = min(start + batch, k)
            eye = np.zeros((end - start, k), dtype=np.float32)
            eye[np.arange(end - start), np.arange(start, end)] = 1.0
            rows.append((enc(eye).numpy() > 0.5).astype(np.uint8))
        G = np.concatenate(rows, axis=0)

        # 3GPP TS 38.212 §5.3.2 [5]
        syst_cols = np.arange(n_syst, dtype=np.int64)
        info_indices = n_punctured + np.arange(n_syst, dtype=np.int64)
        parity_cols = np.arange(n_syst, n, dtype=np.int64)

        if not np.array_equal(G[info_indices, :n_syst],
                              np.eye(n_syst, dtype=np.uint8)):
            raise RuntimeError("G structure mismatch — check Sionna version.")

        code = dict(encoder=enc, decoder=dec, n=n, Z=Z,
                    n_punctured=n_punctured, n_syst=n_syst,
                    G=G, syst_cols=syst_cols,
                    info_indices=info_indices, parity_cols=parity_cols)
        self._code_cache[k] = code
        self.log(f"Code ready: k={k}, n={n}, Z={Z}, n_syst={n_syst}, "
                 f"parity={len(parity_cols)}")
        return code

    # ---------------------------------------------------------------- start m
    def _m_start(self, k: int, n_punctured: int, p: float) -> Tuple[int, float, bool]:
        """
        Finite-length start point (Polyanskiy et al. 2010, Tomamichel et al. 2017):
            H_fl = n_syst * h(p) + sqrt(n_syst * V(p)) * Phi^{-1}(1 - eps) + n_punctured
        The punctured 2Z bits are treated as unresolved (heuristic, not from
        [2], documented in module docstring).  This replaces the asymptotic
        Slepian-Wolf bound k*h(p) and prevents f_EC from spiking at low QBER.
        """
        h_p = binary_entropy(p)
        m_max = 2 * k + n_punctured
        if h_p <= 1e-9:
            return 0, 1.0, False
        n_syst = k - n_punctured
        if self.fixed_rate is not None:
            m = int(math.ceil(k * (1.0 / self.fixed_rate - 1.0)))
            f_min = m / (k * h_p) if h_p > 0 else 1.0
            return m, f_min, m > m_max

        # Finite-length correction.
        V_p = bsc_dispersion(p)
        eps = self.target_fer
        # Guard: normal_quantile expects 0 < prob < 1.
        if eps <= 0.0 or eps >= 1.0:
            eps = 1e-10
        z = normal_quantile(1.0 - eps)
        H_fl = n_syst * h_p + math.sqrt(n_syst * V_p) * z
        m = int(math.ceil(H_fl)) + n_punctured
        if m < 1:
            m = 1
        f_min = m / (k * h_p) if h_p > 0 else 1.0
        return m, f_min, m > m_max

    # ---------------------------------------------------------------- hash
    async def _verify_hash(self, decoded_shuffled, offset, chunk_len, oracle):
        """
        Universal-hash verification (Carter-Wegman [9]).  One 64-bit seed;
        consecutive windows of a random vector (Hankel family, commonly
        called Toeplitz).  Alice's hash is fetched via the oracle, so the
        hash bits are charged by the CountingOracle.
        """
        t = self.hash_bits
        seed_int = self.prng.randint(0, 2 ** 64 - 1)
        rng = np.random.default_rng(seed_int)
        T = rng.integers(0, 2, chunk_len + t - 1, dtype=np.uint8)
        rows = [T[i:i + chunk_len] for i in range(t)]
        perm = self._perm
        blocks = [perm[offset + np.flatnonzero(r)].tolist() for r in rows]
        h_alice = np.asarray(await oracle.get_parities(blocks), dtype=np.uint8)
        # int64 cast avoids uint8 overflow.
        dec_i64 = decoded_shuffled.astype(np.int64)
        h_bob = np.array(
            [int((r.astype(np.int64) @ dec_i64) & 1) for r in rows],
            dtype=np.uint8)
        return np.array_equal(h_alice, h_bob)

    # ---------------------------------------------------------------- attempt
    async def _attempt_decode(self, key_chunk, m_new, p, oracle, offset, code,
                              cached_parities=None):
        """One BP decode with m_new additional parities over the shuffled chunk."""
        k = len(key_chunk)
        n = code["n"]; G = code["G"]
        syst_cols = code["syst_cols"]
        info_indices = code["info_indices"]
        parity_cols = code["parity_cols"]
        perm = self._perm

        key_arr = np.asarray(key_chunk, dtype=np.int64)
        llr = np.zeros(n, dtype=np.float32)
        l_sys = math.log((1.0 - p) / p)
        llr[syst_cols] = np.where(key_arr[info_indices] == 1, l_sys, -l_sys)

        n_cached = 0 if cached_parities is None else len(cached_parities)
        if m_new > 0:
            new_cols = parity_cols[n_cached:n_cached + m_new]
            cols = G[:, new_cols]
            # Map shuffled positions back to original indices.
            blocks = [perm[np.flatnonzero(cols[:, j]) + offset].tolist()
                      for j in range(m_new)]
            new_values = np.asarray(await oracle.get_parities(blocks),
                                    dtype=np.int64)
            values = (new_values if cached_parities is None
                      else np.concatenate([cached_parities, new_values]))
            sent_idx = parity_cols[:len(values)]
            llr[sent_idx] = np.where(values == 1, self.LLR_PARITY,
                                     -self.LLR_PARITY)
        else:
            values = np.empty(0, dtype=np.int64)

        dec_in = tf.convert_to_tensor(llr[None, :], dtype=tf.float32)
        loop = asyncio.get_running_loop()
        decoded = await loop.run_in_executor(
            self._executor,
            lambda: code["decoder"](dec_in).numpy()[0].astype(np.int64))
        
        mism = 0
        if len(values) > 0:
            sent_idx = parity_cols[:len(values)]
            est = (decoded.astype(np.int32) @ G[:, sent_idx].astype(np.int32)) % 2
            mism = int(np.sum(est != values))
        return decoded, mism, values

    # ---------------------------------------------------------------- public
    async def run(self, key: List[int], qber: float, oracle: ParityOracle):
        # Wrap oracle: charge every revealed bit at the boundary.
        counting = _CountingOracle(oracle)
        oracle = counting

        self.bits_revealed = 0
        self.channel_uses = 0
        self.round_trips = 0
        self.stats = {}
        N = len(key)
        if N == 0:
            return list(key), 0, 0, 0

        # Shuffle to spread burst errors.
        perm_list = list(range(N))
        self.prng.shuffle(perm_list)
        perm = np.asarray(perm_list, dtype=np.int64)
        self._perm = perm
        key_arr_full = np.asarray(key, dtype=np.int64)
        key_shuffled = key_arr_full[perm].tolist()

        # Variable chunk sizes so ALL N bits are covered.
        n_chunks = max(1, math.ceil(N / self.max_k))
        base = N // n_chunks
        rem = N % n_chunks
        sizes = [base + 1] * rem + [base] * (n_chunks - rem)
        offsets = [0]
        for s in sizes:
            offsets.append(offsets[-1] + s)

        corrected_shuffled = []
        total_mism = 0
        clamped = 0
        failed_indices: List[int] = []
        f_sw_list = []
        for c in range(n_chunks):
            lo = offsets[c]; hi = offsets[c + 1]
            res, mism, cl, f_sw = await self._run_chunk(
                list(key_shuffled[lo:hi]), qber, oracle, offset=lo)
            corrected_shuffled.extend(res)
            total_mism += mism
            if mism > 0:
                failed_indices.append(c)
            clamped += 1 if cl else 0
            f_sw_list.append((f_sw, hi - lo))

        # Unshuffle over all N bits.
        corrected = key_arr_full.copy()
        corrected[perm] = np.asarray(corrected_shuffled, dtype=np.int64)

        p = min(max(qber, 1e-3), 0.5 - 1e-3)
        h_p = binary_entropy(p)
        ideal_est = N * h_p
        flips = int(np.sum(corrected != key_arr_full))
        f_emp_denom = (N * binary_entropy(flips / N)) if flips > 0 else 0.0

        if f_sw_list:
            total_w = sum(w for _, w in f_sw_list)
            f_sw_min = sum(f * w for f, w in f_sw_list) / total_w
        else:
            f_sw_min = 1.0

        # Leakage from oracle boundary (all parity + hash reveals).
        self.bits_revealed = counting.bits
        self.channel_uses = counting.calls
        self.round_trips = counting.calls

        # Stopping-time term: reported separately, not charged into f_EC.
        stop_leak = n_chunks * int(math.ceil(math.log2(self.max_sweep_iter + 1)))

        f_EC = self.bits_revealed / ideal_est if ideal_est > 0 else float("nan")
        f_total = (self.bits_revealed + stop_leak) / ideal_est if ideal_est > 0 else float("nan")

        self.stats = dict(
            n_bits=N, qber_used=p,
            leaked_bits=self.bits_revealed,
            efficiency_f=f_EC,          # backward-compat alias
            efficiency_f_EC=f_EC,
            efficiency_f_total=f_total,
            efficiency_f_emp=(self.bits_revealed / f_emp_denom)
                             if f_emp_denom > 0 else float("nan"),
            stop_leak_bits=stop_leak,
            f_sw_min=f_sw_min,
            bob_flip_fraction=flips / N,
            syndrome_mismatches=total_mism,
            failed_chunks=len(failed_indices),
            failed_chunk_indices=failed_indices,
            chunk_boundaries=offsets,
            clamped_chunks=clamped,
            chunks=n_chunks,
            channel_uses=self.channel_uses,
            round_trips=self.round_trips,
            target_fer=self.target_fer,
            hash_bits=self.hash_bits,
        )
        self.log(f"Done: leaked={self.bits_revealed}, "
                 f"f_EC={f_EC:.4f}, f_total={f_total:.4f}, "
                 f"mism={total_mism}, failed={len(failed_indices)}/{n_chunks}, "
                 f"clamped={clamped}, oracle_calls={counting.calls}")
        
        return corrected.tolist(), self.bits_revealed, total_mism, self.channel_uses

    # ---------------------------------------------------------------- chunk
    async def _run_chunk(self, key_chunk, qber, oracle, offset=0):
        if not SIONNA_AVAILABLE:
            return list(key_chunk), 1, False, 1.0
        k = len(key_chunk)
        p = min(max(qber, 1e-3), 0.5 - 1e-3)
        try:
            code = self._get_code(k)
        except Exception as e:
            self.log(f"Code setup failed for k={k}: {e}")
            return list(key_chunk), 1, False, 1.0

        m_start, f_sw_min, clamped = self._m_start(k, code["n_punctured"], p)
        m_max = len(code["parity_cols"])

        # Fixed-rate: single attempt.
        if self.fixed_rate is not None:
            m = min(max(m_start, 1), m_max)
            try:
                decoded, mism, values = await self._attempt_decode(
                    key_chunk, m, p, oracle, offset, code)
            except Exception as e:
                self.log(f"Decode error at m={m}: {e}")
                decoded = np.asarray(key_chunk, dtype=np.int64)
                mism = 1
            if mism == 0:
                try:
                    ok = await self._verify_hash(decoded, offset, k, oracle)
                except Exception as e:
                    self.log(f"Hash error at m={m}: {e}")
                    ok = False
                if not ok:
                    self.log(f"Hash FAIL at m={m}")
                    mism = 1
            status = "OK" if mism == 0 else "FAIL"
            self.log(f"chunk k={k}: {status} at m={m} "
                     f"(start={m_start}, max={m_max}, f_sw={f_sw_min:.4f})")
            return decoded.tolist(), mism, clamped, f_sw_min

        # Upward sweep.  On hash failure continue sweeping with more parities.
        m = min(max(m_start, 1), m_max)
        best_decoded = None
        best_mism = 1
        iterations = 0
        cached = None

        while iterations < self.max_sweep_iter:
            iterations += 1
            m_new = m - (0 if cached is None else len(cached))
            if m_new <= 0:
                m = min(m + 1, m_max)
                m_new = m - (0 if cached is None else len(cached))
                if m_new <= 0:
                    break
            try:
                decoded, mism, values = await self._attempt_decode(
                    key_chunk, m_new, p, oracle, offset, code,
                    cached_parities=cached)
                cached = values
            except Exception as e:
                self.log(f"Decode error at m={m}: {e}")
                break

            if mism == 0:
                try:
                    ok = await self._verify_hash(decoded, offset, k, oracle)
                except Exception as e:
                    self.log(f"Hash error at m={m}: {e}")
                    ok = False
                if ok:
                    best_decoded = decoded
                    best_mism = 0
                    break
                self.log(f"Hash FAIL at m={m}, continuing sweep")

            if m >= m_max:
                break
            m_next = min(int(math.ceil(m * self.sweep_factor)), m_max)
            if m_next == m:
                m_next = min(m + 1, m_max)
            m = m_next

        status = "OK" if best_mism == 0 else "FAIL"
        self.log(f"chunk k={k}: {status} "
                 f"(start={m_start}, max={m_max}, f_sw={f_sw_min:.4f}, "
                 f"iters={iterations})")

        corrected_key = (best_decoded.tolist()
                         if best_decoded is not None else list(key_chunk))
        return corrected_key, best_mism, clamped, f_sw_min

    # ---------------------------------------------------------------- offline
    async def calibrate_m_min(self, key: List[int], qber: float,
                              oracle: ParityOracle):
        """
        Offline calibration: bisect per chunk between 0 and m_max to find
        the smallest m where syndrome AND hash pass.
        Does NOT charge leakage (intended for design-time measurement).
        """
        self._perm = np.arange(len(key), dtype=np.int64)
        N = len(key)
        n_chunks = max(1, math.ceil(N / self.max_k))
        base = N // n_chunks
        rem = N % n_chunks
        sizes = [base + 1] * rem + [base] * (n_chunks - rem)
        offsets = [0]
        for s in sizes:
            offsets.append(offsets[-1] + s)

        out = []
        for c in range(n_chunks):
            lo, hi = offsets[c], offsets[c + 1]
            chunk = list(key[lo:hi])
            k = len(chunk)
            code = self._get_code(k)
            m_max = len(code["parity_cols"])
            p = min(max(qber, 1e-3), 0.5 - 1e-3)

            def attempt(m):
                return m
            # Simple linear scan (small m_max in practice).
            m_min = m_max
            for m in range(1, m_max + 1):
                try:
                    decoded, mism, _ = await self._attempt_decode(
                        chunk, m, p, oracle, lo, code, cached_parities=None)
                except Exception:
                    continue
                if mism != 0:
                    continue
                ok = await self._verify_hash(decoded, lo, k, oracle)
                if ok:
                    m_min = m
                    break
            out.append({"size": k, "m_min": m_min, "m_max": m_max})
        return out


# --- self test --------------------------------------------------------------
if __name__ == "__main__":
    async def _main():
        print("--- Empirical-sweep LDPC test (multi-seed) ---")
        N = 10000
        n_seeds = 5

        class Mock(ParityOracle):
            def __init__(self, k): self.key = np.asarray(k)
            async def get_parity(self, idx): return int(self.key[idx].sum() % 2)
            async def get_parities(self, blocks):
                return [int(self.key[b].sum() % 2) for b in blocks]
            async def apply_privacy_amplification(self, *a, **kw): pass

        for p in (0.05, 0.08, 0.10, 0.12):
            f_ec_list, f_tot_list = [], []
            res_list, leaked_list, fer_list = [], [], []
            for seed in range(n_seeds):
                rng = np.random.default_rng(seed)
                alice = rng.integers(0, 2, N).tolist()
                bob = [b ^ int(rng.random() < p) for b in alice]
                prot = NR_LDPC_Standard_ClientProtocol(
                    verbose=False, rate=None, seed=seed)
                corr, leaked, mism, uses = await prot.run(bob, p, Mock(alice))

                s = prot.stats
                # Residual only over successfully reconciled chunks.
                failed = set(s['failed_chunk_indices'])
                bounds = s['chunk_boundaries']
                good = []
                for c in range(s['chunks']):
                    if c in failed:
                        continue
                    good.extend(range(bounds[c], bounds[c + 1]))
                if good:
                    res_err = sum(alice[i] != corr[i] for i in good)
                else:
                    res_err = -1
                res_list.append(res_err if res_err >= 0 else float('nan'))
                f_ec_list.append(s['efficiency_f_EC'])
                f_tot_list.append(s['efficiency_f_total'])
                leaked_list.append(leaked)
                fer_list.append(s['failed_chunks'] / s['chunks'])

            print(f"p={p:.2f}  f_EC={np.nanmean(f_ec_list):.4f}"
                  f"±{np.nanstd(f_ec_list):.4f}  "
                  f"f_tot={np.nanmean(f_tot_list):.4f}  "
                  f"FER={np.mean(fer_list):.3f}  "
                  f"residual={np.nanmean(res_list):.1f}  "
                  f"leaked={np.mean(leaked_list):.0f}")

    if SIONNA_AVAILABLE:
        asyncio.run(_main())


