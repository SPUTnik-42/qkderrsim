# import numpy as np
# import math
# import asyncio
# import os
# import sys
# import time
# from typing import List, Tuple, Optional

# # Add current directory to path
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# from prng import PRNG
# from cascade_refactored import ParityOracle

# # --- GPU / TensorFlow / Sionna Setup ---
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# SIONNA_AVAILABLE = False
# GPU_AVAILABLE = False
# ENCODER_CLASS = None
# DECODER_CLASS = None

# try:
#     import tensorflow as tf
#     # Check GPU
#     gpus = tf.config.list_physical_devices('GPU')
#     if gpus:
#         try:
#             for gpu in gpus:
#                 tf.config.experimental.set_memory_growth(gpu, True)
#             GPU_AVAILABLE = True
#             print(f"[NR-LDPC] GPU Detected: {len(gpus)} device(s). Acceleration ENABLED.")
#         except RuntimeError as e:
#             print(f"[NR-LDPC] GPU Error: {e}")
#     else:
#         print("[NR-LDPC] No GPU detected. Running on CPU (slower).")

#     # Import Sionna (v0.14+ path)
#     from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
#     from sionna.phy.fec.ldpc.decoding import LDPC5GDecoder
    
#     ENCODER_CLASS = LDPC5GEncoder
#     DECODER_CLASS = LDPC5GDecoder
#     SIONNA_AVAILABLE = True
#     print(f"[NR-LDPC] Sionna Library Loaded Successfully.")
    
# except ImportError as e:
#     print(f"[NR-LDPC] Import Error (Sionna/TF): {e}")

# class NR_LDPC_Standard_ClientProtocol:
#     """
#     Standard Client Protocol for 5G NR LDPC Reconciliation.
    
#     Implements 5G Standard-compliant Fixed Rate decoding:
#     - Uses 5G NR LDPC Mother Codes (BG1).
#     - Rate Matching via Transmission of Systematic Bits (known to Bob) + Parity Bits (unknown).
#     - Supports setting specific target rates (default 0.33 for true 5G BG1 rate 1/3).
#     """
#     def __init__(self, verbose=True, rate=0.333, seed=None):
#         """
#         Initialize the protocol.
        
#         Args:
#             verbose (bool): Whether to print logs.
#             rate (float): The target code rate (R = k/n). Lower rate = more parity = better correction.
#                           Standard 5G BG1 base rate is roughly 1/3 (0.33).
#             seed (int): Seed for PRNG.
#         """
#         self.verbose = verbose
#         self.target_rate = rate if rate is not None else 0.333
#         self.prng = PRNG(seed)
#         self.encoder = None
#         self.decoder = None
        
#         # Internal State
#         self.k = 0
#         self.n_mother = 0     # Length of the full mother codeword
#         self.G_matrix = None  # Generator Matrix [k, n_mother] 
#         self.syst_indices = None # Set of indices in codeword corresponding to systematic bits
#         self.out_to_in = {}   # Map: Codeword Index -> Input Key Index
        
#         # Session Stats
#         self.bits_revealed = 0
#         self.channel_uses = 0

#     def log(self, message):
#         if self.verbose:
#             print(f"[NR-LDPC-STD] {message}")

#     def _compute_llr(self, bit, reliability=10.0):
#         # LLR = log(P(1)/P(0))
#         # If bit=1 -> P(1)~1 -> LLR >> 0
#         # If bit=0 -> P(1)~0 -> LLR << 0
#         return reliability if bit == 1 else -reliability

#     def _calculate_puncturing_limit(self, qber: float) -> int:
#         """
#         Calculate how many parity bits are needed based on QBER.
#         Using Shannon Limit as a baseline but strictly sticking to standard rates.
        
#         R <= 1 - H(QBER)
#         n_needed = k / R
        
#         However, we want to be conservative to ensure decoding success.
#         We will use a safety margin (efficiency factor).
#         """
#         if qber <= 0: return 0
#         h_p = -qber * math.log2(qber) - (1-qber) * math.log2(1-qber)
        
#         # Theoretical max rate
#         max_rate = 1.0 - h_p
        
#         # Avoid division by zero close to 1
#         if max_rate < 0.05: max_rate = 0.05
        
#         # Calculate N needed to satisfy this rate
#         n_needed = int(self.k / max_rate)
        
#         # Parity bits needed = n_needed - k_systematic (approx)
#         # To add a safety margin so decoding doesn't mostly fail, we inflate the needed parity.
#         margin = 1.2  # Safety margin multiplier (30% more parity over base computation)
#         parity_needed = int(max(0, n_needed - self.k) * margin)
        
#         # Ensure we don't request more than available in the mother code
#         # The mother code has (n_mother - k) parity bits roughly.
        
#         # Available parity indices
#         available_parity = self.n_mother - len(self.syst_indices)
#         if parity_needed > available_parity:
#             parity_needed = available_parity
            
#         return parity_needed

#     async def run(self, key: List[int], qber: float, oracle: ParityOracle):
#         """
#         Execute Reconciliation with automatic block chunking for large keys.
#         Using safe MAX_K=4000 to avoid Sionna GPU reshape instability at very large block sizes.
#         """
#         MAX_K = 4000
#         if len(key) <= MAX_K:
#             return await self._run_chunk(key, qber, oracle, 0)
            
#         self.log(f"Key length {len(key)} exceeds max LDPC block size {MAX_K}. Splitting into chunks.")
        
#         results = []
#         total_bits = 0
#         total_errors = 0
#         total_channel = 0
        
#         for i in range(0, len(key), MAX_K):
#             chunk = key[i:i+MAX_K]
#             res_chunk, bits, errors, ch_uses = await self._run_chunk(chunk, qber, oracle, offset=i)
#             results.extend(res_chunk)
#             total_bits += bits
#             total_errors += errors
#             total_channel += ch_uses
            
#         return results, total_bits, total_errors, total_channel

#     async def _run_chunk(self, key: List[int], qber: float, oracle: ParityOracle, offset: int = 0):
#         """
#         Execute Reconciliation in one shot with Rate Matching (Puncturing) for a single block.
#         """
#         if not SIONNA_AVAILABLE:
#             self.log("CRITICAL: Sionna not available. Aborting.")
#             return key, 0, 0, 0
            
#         key_array = np.array(key, dtype=int)
#         n_key = len(key_array)
#         self.k = n_key
        
#         if qber <= 0: qber = 0.001
        
#         self.bits_revealed = 0
#         self.channel_uses = 0
        
#         # 1. Initialize 5G Encoder (Mother Code) - ALWAYS USE STANDARD LOW RATE (High Reliability)
#         # We initialize the encoder with the default base rate (e.g. 1/3) to have max parity available.
#         # Then we puncture (don't send) based on QBER.
#         # If user specified a specific hard-coded rate in __init__, we respect it as the "Mother Code" limit,
#         # but we still puncture if QBER allows.
        
#         # Standard BG1 is Rate 1/3 => n = 3k
#         # Use the configured target_rate to set the MOTHER code dimension.
#         mother_n = int(self.k / self.target_rate)
        
#         # Enforce Sionna's limit: coderate >= 1/3 (therefore n <= 3k)
#         if mother_n > 3 * self.k:
#             mother_n = 3 * self.k
            
#         self.log(f"Initializing 5G LDPC Engine (k={self.k}, Mother n={mother_n}, Base Rate={self.target_rate:.3f})...")
        
#         try:
#             # Re-init if parameters changed
#             if self.encoder is None or self.encoder.k != self.k or self.encoder._n != mother_n:
#                 self.encoder = ENCODER_CLASS(k=self.k, n=mother_n)
#                 # Standard decoding iterations
#                 self.decoder = DECODER_CLASS(self.encoder, num_iter=20, return_infobits=True)
                
#                 # Check actual n
#                 dummy = np.zeros((1, self.k), dtype=np.float32)
#                 res = self.encoder(dummy)
#                 self.n_mother = res.shape[1]
#                 self.log(f"Code Configured: n={self.n_mother} (Actual Rate {self.k/self.n_mother:.3f})")
                
#                 # Invalidate G Cache
#                 self.G_matrix = None
#         except Exception as e:
#             self.log(f"Setup Error: {e}")
#             return key, 0, 0, 0

#         # 2. Compute/Cache Generator Matrix G
#         # We need this to simulate Oracle responses (mapping Parity Check to Bit Value).
#         if self.G_matrix is None:
#             self.log("Computing Generator Matrix G...")
#             try:
#                 # Use batches for simplicity
#                 batch_size = 256 # Standard batch size
#                 G_rows = []
#                 num_batches = math.ceil(self.k / batch_size)
                
#                 # We encode the Identity Matrix row by row (batched)
#                 # Row i of I -> Encoded -> Row i of G
#                 for i in range(num_batches):
#                     start = i * batch_size
#                     end = min(start + batch_size, self.k)
#                     size = end - start
                    
#                     batch_in = np.zeros((size, self.k), dtype=np.float32)
#                     for idx in range(size):
#                         batch_in[idx, start + idx] = 1.0
                    
#                     # Ensure encoder can handle batch
#                     cw = self.encoder(batch_in).numpy()
#                     G_rows.append(cw)
                
#                 self.G_matrix = np.concatenate(G_rows, axis=0) # [k, n]
#                 self.G_matrix = (self.G_matrix > 0.5).astype(int)
                
#                 # Analyze Structure (Systematic vs Parity)
#                 col_weights = np.sum(self.G_matrix, axis=0)
#                 syst_cols = np.where(col_weights == 1)[0]
#                 self.syst_indices = set(syst_cols)
                
#                 self.out_to_in = {}
#                 for col in syst_cols:
#                     row = np.where(self.G_matrix[:, col] == 1)[0][0]
#                     self.out_to_in[col] = row
                    
#                 self.log(f"G Matrix Ready. Systematic Bits: {len(self.syst_indices)}")
                
#             except Exception as e:
#                 self.log(f"G Matrix Computation Failed: {e}")
#                 return key, 0, 0, 0

#         # 3. One-Shot Punctured Exchange
        
#         # Buffer for LLRs
#         # Initialize with 0s. 0 means "No Information" (Punctured).
#         llr_buffer = np.zeros(self.n_mother, dtype=np.float32)
        
#         # Fill Systematic Information (known to Bob, with noise)
#         syst_llr_mag = math.log((1.0 - qber)/qber)
        
#         for out_idx, in_idx in self.out_to_in.items():
#             bit_val = key_array[in_idx]
#             llr_buffer[out_idx] = self._compute_llr(bit_val, reliability=syst_llr_mag)
            
#         # Identify available parity bits
#         available_parity_indices = []
#         for j in range(self.n_mother):
#             if j not in self.syst_indices:
#                 available_parity_indices.append(j)
        
#         # --- RATE MATCHING / PUNCTURING LOGIC ---
#         # Calculate how many parity bits we actually need based on QBER
#         # We start filling the circular buffer (linear in available_parity_indices)
#         # up to the calculated limit.
        
#         num_parity_to_send = self._calculate_puncturing_limit(qber)
        
#         self.log(f"Rate Matching: QBER={qber:.4f} => Need ~{num_parity_to_send} parity bits (out of {len(available_parity_indices)} available).")
        
#         # Select the bits (Standard: Linear selection from start of circular buffer)
#         # In 5G LDPC, the first 2*Zc columns are punctured, but Sionna handles the output map.
#         # We just take the first N bits from the encoder output that are parity.
        
#         parity_indices_to_request = available_parity_indices[:num_parity_to_send]
        
#         # Request Selected Parity Bits
#         if parity_indices_to_request:
#             blocks = []
#             for p_idx in parity_indices_to_request:
#                 participants = np.where(self.G_matrix[:, p_idx] == 1)[0].tolist()
#                 blocks.append([idx + offset for idx in participants])
            
#             self.log(f"Requesting {len(blocks)} parity bits (Punctured Transmission).")
#             self.channel_uses += 1
            
#             try:
#                 # Async Request
#                 values = await oracle.get_parities(blocks)
#                 self.bits_revealed += len(values)
                
#                 # Update LLRs for received bits
#                 for p_idx, val in zip(parity_indices_to_request, values):
#                     llr_buffer[p_idx] = self._compute_llr(val, reliability=100.0)
                    
#             except Exception as e:
#                 self.log(f"Oracle Error: {e}")
#                 return key, 0, 0, 0
#         else:
#             self.log("No parity needed (QBER extremely low).")

#         # 4. Decode
#         try:
#             dc_in = tf.convert_to_tensor([llr_buffer], dtype=tf.float32)
#             decoded_est = self.decoder(dc_in).numpy()[0].astype(int)
            
#             # Check correctness (Verify parity constraints)
#             est_codeword = np.dot(decoded_est, self.G_matrix) % 2
            
#             errors = 0
#             # Verify against requested parities
#             # Note: We can only verify against bits we actually received!
#             if parity_indices_to_request:
#                 for p_idx, val in zip(parity_indices_to_request, values):
#                     if est_codeword[p_idx] != val:
#                         errors += 1
            
#             # Check against systematic (if we had access to real key, but here we don't know real key)
#             # This 'errors' is just Syndrome Check on the received vector.
#             # If rate is high (heavy puncturing), we might pass syndrome but fail decoding if not enough info.
#             # But the Parity Oracle returns true parity.
            
#             if errors == 0:
#                 self.log("Decoding Success! Parity checks passed.")
#             else:
#                 self.log(f"Decoding Failed: {errors} parity mismatches.")
                
#             return decoded_est.tolist(), self.bits_revealed, errors, self.channel_uses
                    
#         except Exception as e:
#             self.log(f"Decoding Error: {e}")
#             return key, self.bits_revealed, 0, self.channel_uses

# if __name__ == "__main__":
#     async def main():
#         print("--- 5G NR LDPC Standard Test ---")
#         N = 10000
#         prng = PRNG(123)
#         alice = [prng.randint(0,1) for _ in range(N)]
#         qber = 0.1
#         bob = list(alice)
#         errs = 0
#         for i in range(N):
#             if prng.random() < qber:
#                 bob[i] = 1 - bob[i]
#                 errs += 1
        
#         print(f"Details: N={N}, QBER={qber}, Errors={errs}")

#         class MockOracle:
#             def __init__(self, k): self.k = np.array(k)
#             async def get_parities(self, blocks):
#                 return [int(np.sum(self.key[b])%2) for b in blocks]
        
#         oracle = MockOracle(alice)
#         oracle.key = np.array(alice)
        
#         # Test 
#         prot = NR_LDPC_Standard_ClientProtocol(verbose=True, rate=0.333)
#         res, bits, fail, _ = await prot.run(bob, qber, oracle)
        
#         final_errs = np.sum(np.abs(np.array(res) - np.array(alice)))
#         print(f"Final Errors: {final_errs}")
#         print(f"Bits Revealed: {bits}")

#     if SIONNA_AVAILABLE:
#         asyncio.run(main())







# """
# Plain puncturing-only 5G NR LDPC reconciliation for BB84.

# Rate selection uses the Polyanskiy-Poor-Verdu (PPV) finite-blocklength
# normal approximation.  The reconciliation efficiency f is measured from
# the number of parity bits actually revealed; it is NEVER an input.

# References
# ----------
# [1] Slepian & Wolf, "Noiseless coding of correlated information sources",
#     IEEE Trans. Inf. Theory 19(4), 1973.
#         m >= k * h(p).
# [2] Polyanskiy, Poor, Verdu, "Channel Coding Rate in the Finite Blocklength
#     Regime", IEEE Trans. Inf. Theory 56(5), 2307-2347, 2010.
#         R = 1 - h(p) - sqrt(V(p)/k) * Q^{-1}(eps),
#         V(p) = p(1-p) [log2((1-p)/p)]^2.
# [3] Elkouss, Martinez-Mateo, Martin, Quantum Inf. Comput. 11(3), 2011.
#         f = m / (k * h(p)).
# [4] Richardson & Urbanke, Modern Coding Theory, Cambridge, 2008.
#         BSC(p) LLR (Sionna convention ln P(1)/P(0)) = +-ln((1-p)/p);
#         punctured bits: LLR = 0; noiseless bits: LLR ~ finite large.
# [5] 3GPP TS 38.212 §5.4.2  (5G NR LDPC rate matching).
#         E = k/R transmitted bits for target rate R  =>  m = ceil(k/R) - k.
# [6] Gallager, Low-Density Parity-Check Codes, MIT Press, 1963.
# """

# import asyncio
# import math
# import os
# import sys
# from typing import Dict, List, Optional, Tuple

# import numpy as np

# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# from prng import PRNG
# from cascade_refactored import ParityOracle

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# SIONNA_AVAILABLE = False
# GPU_AVAILABLE = False

# try:
#     import tensorflow as tf
#     gpus = tf.config.list_physical_devices('GPU')
#     if gpus:
#         try:
#             for gpu in gpus:
#                 tf.config.experimental.set_memory_growth(gpu, True)
#             GPU_AVAILABLE = True
#             print(f"[LDPC] GPU detected: {len(gpus)} device(s).")
#         except RuntimeError as e:
#             print(f"[LDPC] GPU setup warning: {e}")
#     else:
#         print("[LDPC] No GPU detected; running on CPU.")

#     from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
#     from sionna.phy.fec.ldpc.decoding import LDPC5GDecoder

#     SIONNA_AVAILABLE = True
#     print("[LDPC] Sionna loaded successfully.")
# except ImportError as e:
#     print(f"[LDPC] Sionna/TF import error: {e}")


# # --- Helpers -------------------------------------------------------------------
# def binary_entropy(p: float) -> float:
#     """h(p) = -p log2 p - (1-p) log2(1-p); 0 at p in {0,1}."""
#     if p <= 0.0 or p >= 1.0:
#         return 0.0
#     return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


# def bsc_dispersion(p: float) -> float:
#     """
#     Channel dispersion V(p) for the binary symmetric channel,
#     PPV 2010, IEEE Trans. Inf. Theory 56(5), §III-B.
#         V(p) = p(1-p) * [log2((1-p)/p)]^2
#     """
#     p = min(max(p, 1e-6), 1.0 - 1e-6)
#     return p * (1.0 - p) * (math.log2((1.0 - p) / p)) ** 2


# def qfunc_inv(x: float) -> float:
#     """
#     Inverse of the standard Gaussian tail Q(x) = (1/2) erfc(x/sqrt(2)).
#     Uses the rational approximation of Acklam (2003), "An algorithm for
#     numerical inversion of the normal CDF", accurate to ~1e-9 for
#     x in (0, 1).  Standard tool in finite-blocklength coding theory.
#     """
#     x = min(max(x, 1e-15), 1.0 - 1e-15)
#     # Peter Acklam's inverse normal CDF
#     a = [-3.969683028665376e+01,  2.209460984245205e+02,
#          -2.759285104469687e+02,  1.383577518672690e+02,
#          -3.066479806614716e+01,  2.506628277459239e+00]
#     b = [-5.447609879822406e+01,  1.615858368580409e+02,
#          -1.556989798598866e+02,  6.680131188771972e+01,
#          -1.328068155288572e+01]
#     c = [-7.784894002430293e-03, -3.223964580411365e-01,
#          -2.400758277161838e+00, -2.549732539343734e+00,
#           4.374664141464968e+00,  2.938163982698783e+00]
#     d = [ 7.784695709041462e-03,  3.224671290700398e-01,
#           2.445134137142996e+00,  3.754408661907416e+00]
#     plow, phigh = 0.02425, 1.0 - 0.02425
#     if x < plow:
#         q = math.sqrt(-2.0 * math.log(x))
#         return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
#                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
#     if x > phigh:
#         q = math.sqrt(-2.0 * math.log(1.0 - x))
#         return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
#                 ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
#     q = x - 0.5
#     r = q * q
#     return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
#            (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)


# # --- Protocol ------------------------------------------------------------------
# class NR_LDPC_Standard_ClientProtocol:
#     """
#     Puncturing-only 5G NR LDPC reconciliation.

#     Parameters
#     ----------
#     verbose     : print progress.
#     rate        : if None (default), rate is derived per-chunk from the apriori
#                   QBER using the PPV finite-blocklength normal approximation
#                   [2].  If a float in [1/3, 1) is supplied, that value is used
#                   as a FIXED target rate (no QBER adaptation) — useful for
#                   controlled ablation studies.
#     seed        : PRNG seed (interface compatibility).
#     num_iter    : BP iterations per decode.
#     max_k       : chunk size; keys longer than this are split into equal blocks.
#     target_fer  : target block failure probability eps in the PPV formula [2].
#                   Default 1e-3, matching the QKD reconciliation literature
#                   (Elkouss et al. 2011, §III).  This is a reliability spec,
#                   NOT a reconciliation-efficiency knob.
#     """

#     LLR_PARITY = 100.0

#     def __init__(
#         self,
#         verbose: bool = True,
#         rate: Optional[float] = None,
#         seed: Optional[int] = None,
#         num_iter: int = 20,
#         max_k: int = 4000,
#         target_fer: float = 1e-3,
#     ):
#         self.verbose = verbose
#         if rate is None:
#             self.fixed_target_rate = None
#         else:
#             # Only treat > 1/3 as an explicit fixed rate.  The BobClient
#             # default 0.333 is interpreted as "auto" so this class does the
#             # right thing without any callsite changes.
#             r = float(rate)
#             self.fixed_target_rate = r if r > 1.0/3.0 + 1e-3 else None
#         self.num_iter = int(num_iter)
#         self.max_k = int(max_k)
#         self.target_fer = float(target_fer)
#         self.prng = PRNG(seed)

#         self._code_cache: Dict[int, dict] = {}

#         self.bits_revealed = 0
#         self.channel_uses = 0
#         self.stats: Dict[str, float] = {}

#     def log(self, msg: str) -> None:
#         if self.verbose:
#             print(f"[LDPC] {msg}")

#     def log_metrics(self) -> None:
#         if not self.stats:
#             return
#         s = self.stats
#         self.log(
#             f"Metrics: N={s['n_bits']}, qber={s['qber_used']:.4f}, "
#             f"leaked={s['leaked_bits']}, f={s['efficiency_f']:.4f}, "
#             f"f_pred_ppv={s['f_predicted_ppv']:.4f}, "
#             f"residual_qber={s['realized_qber']:.4f}, "
#             f"mismatches={s['syndrome_mismatches']}"
#         )

#     # ------------------------------------------------------------- code setup
#     def _get_code(self, k: int) -> dict:
#         if k in self._code_cache:
#             return self._code_cache[k]
#         if not SIONNA_AVAILABLE:
#             raise RuntimeError("Sionna/TF unavailable: cannot construct LDPC code.")

#         n_mother = 3 * k
#         encoder = LDPC5GEncoder(k=k, n=n_mother)
#         decoder = LDPC5GDecoder(encoder, num_iter=self.num_iter, return_infobits=True)
#         n = int(getattr(encoder, "n", getattr(encoder, "_n", n_mother)))

#         batch = 256
#         rows = []
#         for start in range(0, k, batch):
#             end = min(start + batch, k)
#             eye = np.zeros((end - start, k), dtype=np.float32)
#             eye[np.arange(end - start), np.arange(start, end)] = 1.0
#             rows.append((encoder(eye).numpy() > 0.5).astype(np.uint8))
#         G = np.concatenate(rows, axis=0)

#         col_weights = G.sum(axis=0)
#         syst_cols = np.flatnonzero(col_weights == 1)
#         syst_rows = np.argmax(G[:, syst_cols], axis=0)
#         parity_cols = np.flatnonzero(col_weights != 1)

#         code = dict(encoder=encoder, decoder=decoder, n=n, G=G,
#                     syst_cols=syst_cols, syst_rows=syst_rows,
#                     parity_cols=parity_cols)
#         self._code_cache[k] = code
#         self.log(f"Code ready: k={k}, n={n}, R0={k/n:.4f}, "
#                  f"systematic={len(syst_cols)}, parity={len(parity_cols)}")
#         return code

#     # ------------------------------------------------------------- rate math
#     def _target_rate_ppv(self, k: int, p: float) -> float:
#         """
#         PPV finite-blocklength rate for a BSC(p) at block length k and
#         target FER eps [2]:
#             R = 1 - h(p) - sqrt(V(p)/k) * Q^{-1}(eps)
#         Result is clamped to the mother code's [1/3, 1) range.
#         """
#         h_p = binary_entropy(p)
#         V_p = bsc_dispersion(p)
#         backoff = math.sqrt(V_p / k) * qfunc_inv(self.target_fer)
#         R = 1.0 - h_p - backoff
#         return min(max(R, 1.0 / 3.0 + 1e-6), 1.0 - 1e-9)

#     def _target_rate(self, k: int, qber: float) -> float:
#         if self.fixed_target_rate is not None:
#             return self.fixed_target_rate
#         return self._target_rate_ppv(k, qber)

#     # ------------------------------------------------------------- public API
#     async def run(self, key: List[int], qber: float, oracle: ParityOracle):
#         self.bits_revealed = 0
#         self.channel_uses = 0
#         self.stats = {}

#         N = len(key)
#         if N == 0:
#             return list(key), 0, 0, 0

#         n_chunks = max(1, math.ceil(N / self.max_k))
#         if n_chunks > 1:
#             self.log(f"Key length {N} > max_k={self.max_k}: {n_chunks} equal chunks.")
#         bounds = np.linspace(0, N, n_chunks + 1).astype(int)

#         corrected: List[int] = []
#         total_mismatches = 0
#         failed_chunks = 0
#         for c in range(n_chunks):
#             lo, hi = int(bounds[c]), int(bounds[c + 1])
#             res, mism = await self._run_chunk(list(key[lo:hi]), qber, oracle, offset=lo)
#             corrected.extend(res)
#             total_mismatches += mism
#             failed_chunks += 1 if mism > 0 else 0

#         # --- measured post-hoc stats [1][3] ---
#         p = min(max(qber, 1e-3), 0.5 - 1e-3)
#         ideal_leak = N * binary_entropy(p)
#         flips = int(np.sum(np.asarray(corrected) != np.asarray(key)))
#         realized_q = flips / N
#         ideal_real = N * binary_entropy(realized_q)

#         # PPV predicted leakage for the whole block, using per-chunk k
#         k_chunk = max(1, min(N, self.max_k))
#         R_ppv = self._target_rate(k_chunk, p)
#         m_ppv = max(0.0, k_chunk / R_ppv - k_chunk) * n_chunks
#         f_pred = m_ppv / ideal_leak if ideal_leak > 0 else float("nan")

#         self.stats = dict(
#             n_bits=N,
#             qber_used=p,
#             leaked_bits=self.bits_revealed,
#             efficiency_f=(self.bits_revealed / ideal_leak
#                           if ideal_leak > 0 else float("nan")),
#             f_predicted_ppv=f_pred,
#             realized_qber=realized_q,
#             efficiency_f_realized=(self.bits_revealed / ideal_real
#                                    if ideal_real > 0 else float("nan")),
#             syndrome_mismatches=total_mismatches,
#             failed_chunks=failed_chunks,
#             chunks=n_chunks,
#             channel_uses=self.channel_uses,
#             target_fer=self.target_fer,
#         )
#         self.log(
#             f"Done: leaked={self.bits_revealed}, f_meas={self.stats['efficiency_f']:.4f}, "
#             f"f_ppv={f_pred:.4f}, mism={total_mismatches}, failed={failed_chunks}/{n_chunks}"
#         )
#         return corrected, self.bits_revealed, total_mismatches, self.channel_uses

#     # ------------------------------------------------------------- chunk
#     async def _run_chunk(
#         self, key: List[int], qber: float, oracle: ParityOracle, offset: int = 0,
#     ) -> Tuple[List[int], int]:
#         if not SIONNA_AVAILABLE:
#             self.log("Sionna unavailable; returning key unchanged.")
#             return list(key), 1

#         k = len(key)
#         key_arr = np.asarray(key, dtype=np.int64)
#         p = min(max(qber, 1e-3), 0.5 - 1e-3)

#         try:
#             code = self._get_code(k)
#         except Exception as e:
#             self.log(f"Code setup failed: {e}")
#             return list(key), 1

#         n = code["n"]
#         G = code["G"]
#         parity_cols = code["parity_cols"]
#         n_parity_avail = len(parity_cols)

#         # ---- target rate from PPV or explicit override [2] ----
#         R = self._target_rate(k, p)
#         m_theory = int(math.ceil(k / R - k))       # E = k/R per [5]
#         m = max(0, min(m_theory, n_parity_avail))
#         sent_idx = parity_cols[:m]

#         self.log(f"chunk k={k}: R_target={R:.4f} -> m={m} parity "
#                  f"(available={n_parity_avail}, punctured={n_parity_avail - m})")

#         # ---- LLR buffer [4] ----
#         llr = np.zeros(n, dtype=np.float32)
#         l_sys = math.log((1.0 - p) / p)
#         llr[code["syst_cols"]] = np.where(
#             key_arr[code["syst_rows"]] == 1, l_sys, -l_sys
#         )

#         values = np.empty(0, dtype=np.int64)
#         if m > 0:
#             cols = G[:, sent_idx]
#             blocks = [(np.flatnonzero(cols[:, j]) + offset).tolist() for j in range(m)]
#             self.channel_uses += 1
#             try:
#                 values = np.asarray(await oracle.get_parities(blocks), dtype=np.int64)
#             except Exception as e:
#                 self.log(f"Oracle error: {e}")
#                 return list(key), 1
#             self.bits_revealed += len(values)
#             llr[sent_idx] = np.where(values == 1, self.LLR_PARITY, -self.LLR_PARITY)

#         # ---- single-shot BP decode [6] ----
#         try:
#             dec_in = tf.convert_to_tensor(llr[None, :], dtype=tf.float32)
#             decoded = code["decoder"](dec_in).numpy()[0].astype(np.int64)
#         except Exception as e:
#             self.log(f"Decode error: {e}")
#             return list(key), 1

#         mismatches = 0
#         if m > 0:
#             est = (decoded.astype(np.int32) @ G[:, sent_idx].astype(np.int32)) % 2
#             mismatches = int(np.sum(est != values))

#         corrected_key = np.zeros(k, dtype=np.int64)
#         corrected_key[code["syst_rows"]] = decoded[code["syst_cols"]]

#         self.log("Decode OK." if mismatches == 0
#                  else f"Decode FAILED: {mismatches} parity mismatches.")
#         return corrected_key.tolist(), mismatches


# if __name__ == "__main__":

#     async def _main():
#         print("--- PPV-rate plain puncturing LDPC reconciliation test ---")
#         N, qber = 10000, 0.05
#         rng = np.random.default_rng(123)
#         alice = rng.integers(0, 2, N).tolist()
#         bob = list(alice)
#         for i in range(N):
#             if rng.random() < qber:
#                 bob[i] ^= 1
#         print(f"N={N}, true qber={qber}")

#         class MockOracle(ParityOracle):
#             def __init__(self, key): self.key = np.asarray(key)
#             async def get_parity(self, indices):
#                 return int(self.key[indices].sum() % 2)
#             async def get_parities(self, blocks):
#                 return [int(self.key[b].sum() % 2) for b in blocks]
#             async def apply_privacy_amplification(self, *a, **kw):
#                 pass

#         oracle = MockOracle(alice)

#         for eps in (1e-2, 1e-3, 1e-4):
#             prot = NR_LDPC_Standard_ClientProtocol(
#                 verbose=False, rate=None, seed=42, target_fer=eps
#             )
#             corrected, leaked, mism, uses = await prot.run(bob, qber, oracle)
#             residual = sum(a != b for a, b in zip(alice, corrected))
#             print(f"eps={eps:.0e}  leaked={leaked}  f_meas="
#                   f"{prot.stats['efficiency_f']:.4f}  f_ppv="
#                   f"{prot.stats['f_predicted_ppv']:.4f}  "
#                   f"residual={residual}/{N}  failed={prot.stats['failed_chunks']}")

#     if SIONNA_AVAILABLE:
#         asyncio.run(_main())

# """
# Puncturing-only 5G NR LDPC reconciliation for BB84, with empirically
# determined parity requirement per chunk and universal-hash verification.

# Leakage is measured at the oracle boundary via a CountingOracle wrapper.

# Reported efficiencies:
#     f_EC    = leaked_bits / (k*h(p))
#     f_total = (leaked_bits + stop_leak) / (k*h(p))

# References
# ----------
# [1] Slepian & Wolf, Bell System Technical Journal 52(10):1657-1687, 1973.
# [2] Tan & Kosut, IEEE Trans. Inf. Theory 60(2):881-903, 2014.
# [3] Polyanskiy, Poor, Verdu, IEEE Trans. Inf. Theory 56(5):2307-2359, 2010.
# [4] Elkouss, Martinez-Mateo, Martin, Quantum Inf. Comput. 11(3):226-238, 2011.
# [5] 3GPP TS 38.212 §5.3.2 (first 2Z systematic positions punctured).
# [6] Richardson & Urbanke, Modern Coding Theory, Cambridge, 2008.
# [7] Gallager, Low-Density Parity-Check Codes, MIT Press, 1963.
# [8] Sionna LDPC5GDecoder: llr_max default is 20.0.
# [9] Carter & Wegman, J. Comput. Syst. Sci. 18(2):143-154, 1979.
# [10] Martinez-Mateo, Elkouss, Martin, Quantum Inf. Comput. 13(9):791-812, 2013.
#      Rate-adaptive reconciliation starts at m_start = k * h(p).
# """

# import asyncio, math, os, sys
# from concurrent.futures import ThreadPoolExecutor
# from typing import Dict, List, Optional, Tuple
# import numpy as np

# # --- local imports only (never from bb84_server_client) ---
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# from prng import PRNG
# from cascade_refactored import ParityOracle

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# SIONNA_AVAILABLE = False

# try:
#     import tensorflow as tf
#     gpus = tf.config.list_physical_devices('GPU')
#     if gpus:
#         for g in gpus:
#             tf.config.experimental.set_memory_growth(g, True)
#     from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
#     from sionna.phy.fec.ldpc.decoding   import LDPC5GDecoder
#     SIONNA_AVAILABLE = True
#     print("[LDPC] Sionna loaded.")
# except ImportError as e:
#     print(f"[LDPC] Sionna/TF import error: {e}")
# except Exception as e:
#     print(f"[LDPC] Unexpected error loading Sionna/TF: {e}")


# # --- math helpers -----------------------------------------------------------
# def binary_entropy(p: float) -> float:
#     """h(p) = -p log2 p - (1-p) log2(1-p)."""
#     if p <= 0.0 or p >= 1.0:
#         return 0.0
#     return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


# # --- leakage-counting wrapper ----------------------------------------------
# class _CountingOracle(ParityOracle):
#     def __init__(self, wrapped: ParityOracle):
#         self.wrapped = wrapped
#         self.bits = 0
#         self.calls = 0

#     async def get_parity(self, indices: List[int]) -> int:
#         r = await self.wrapped.get_parity(indices)
#         self.bits += 1
#         self.calls += 1
#         return r

#     async def get_parities(self, blocks: List[List[int]]) -> List[int]:
#         r = await self.wrapped.get_parities(blocks)
#         self.bits += len(r)
#         self.calls += 1
#         return r

#     async def apply_privacy_amplification(self, *a, **kw):
#         return await self.wrapped.apply_privacy_amplification(*a, **kw)


# # --- protocol ---------------------------------------------------------------
# class NR_LDPC_Standard_ClientProtocol:
#     """
#     Puncturing-only 5G NR LDPC reconciliation. See module docstring.
#     """

#     LLR_PARITY = 100.0

#     def __init__(self, verbose=True, rate=None, seed=None,
#                  num_iter=50, max_k=4000, target_fer=1e-10,
#                  sweep_factor=1.05, max_sweep_iter=100,
#                  hash_bits=64):
#         self.verbose = verbose
#         if rate is None or float(rate) <= 1.0/3.0 + 1e-3:
#             self.fixed_rate: Optional[float] = None
#         else:
#             self.fixed_rate = float(rate)
#         self.num_iter = int(num_iter)
#         self.max_k = int(max_k)
#         self.target_fer = float(target_fer)
#         self.sweep_factor = max(1.005, float(sweep_factor))
#         self.max_sweep_iter = int(max_sweep_iter)
#         self.hash_bits = int(hash_bits)
#         self.prng = PRNG(seed)
#         self._code_cache: Dict[int, dict] = {}
#         self._perm: Optional[np.ndarray] = None
#         self.bits_revealed = 0
#         self.channel_uses = 0
#         self.round_trips = 0
#         self.stats: Dict[str, float] = {}
#         self._executor = ThreadPoolExecutor(max_workers=1)

#     def log(self, msg):
#         if self.verbose:
#             print(f"[LDPC] {msg}")

#     # ---------------------------------------------------------------- code
#     def _get_code(self, k: int) -> dict:
#         if k in self._code_cache:
#             return self._code_cache[k]
#         if not SIONNA_AVAILABLE:
#             raise RuntimeError("Sionna/TF unavailable.")

#         n_mother = 3 * k
#         enc = LDPC5GEncoder(k=k, n=n_mother)
#         dec = LDPC5GDecoder(enc, num_iter=self.num_iter,
#                             return_infobits=True,
#                             llr_max=self.LLR_PARITY)
#         n = int(getattr(enc, "n", getattr(enc, "_n", n_mother)))
#         Z = int(getattr(enc, "_z", getattr(enc, "z", 0)))
#         if Z <= 0:
#             raise RuntimeError(f"Could not determine Z for k={k}.")
#         n_punctured = 2 * Z
#         n_syst = k - n_punctured
#         if n_syst <= 0:
#             raise RuntimeError(f"k={k} too small for Z={Z}.")

#         batch = 256
#         rows = []
#         for start in range(0, k, batch):
#             end = min(start + batch, k)
#             eye = np.zeros((end - start, k), dtype=np.float32)
#             eye[np.arange(end - start), np.arange(start, end)] = 1.0
#             rows.append((enc(eye).numpy() > 0.5).astype(np.uint8))
#         G = np.concatenate(rows, axis=0)

#         syst_cols = np.arange(n_syst, dtype=np.int64)
#         info_indices = n_punctured + np.arange(n_syst, dtype=np.int64)
#         parity_cols = np.arange(n_syst, n, dtype=np.int64)

#         if not np.array_equal(G[info_indices, :n_syst],
#                               np.eye(n_syst, dtype=np.uint8)):
#             raise RuntimeError("G structure mismatch — check Sionna version.")

#         code = dict(encoder=enc, decoder=dec, n=n, Z=Z,
#                     n_punctured=n_punctured, n_syst=n_syst,
#                     G=G, syst_cols=syst_cols,
#                     info_indices=info_indices, parity_cols=parity_cols)
#         self._code_cache[k] = code
#         self.log(f"Code ready: k={k}, n={n}, Z={Z}, n_syst={n_syst}, "
#                  f"parity={len(parity_cols)}")
#         return code

#     # ---------------------------------------------------------------- start m
#     def _m_start(self, k: int, n_punctured: int, p: float) -> Tuple[int, float, bool]:
#         """
#         Slepian-Wolf entropy lower bound as sweep start point.

#         H(X|Y) = k * h(p) for the full key.  Punctured systematic bits are
#         part of Bob's sifted key — his information about them is the same
#         BSC channel as the un-punctured bits, so their conditional entropy
#         is h(p), not 1.

#         Reference: Slepian & Wolf (1973); Martinez-Mateo et al. (2013).
#         """
#         h_p = binary_entropy(p)
#         m_max = 2 * k + n_punctured
#         if h_p <= 1e-9:
#             return 0, 1.0, False
#         if self.fixed_rate is not None:
#             m = int(math.ceil(k * (1.0 / self.fixed_rate - 1.0)))
#             f_min = m / (k * h_p) if h_p > 0 else 1.0
#             return m, f_min, m > m_max
#         m = int(math.ceil(k * h_p))
#         f_min = m / (k * h_p)
#         return m, f_min, m > m_max

#     # ---------------------------------------------------------------- hash
#     async def _verify_hash(self, decoded_shuffled, offset, chunk_len, oracle):
#         t = self.hash_bits
#         seed_int = self.prng.randint(0, 2 ** 64 - 1)
#         rng = np.random.default_rng(seed_int)
#         T = rng.integers(0, 2, chunk_len + t - 1, dtype=np.uint8)
#         rows = [T[i:i + chunk_len] for i in range(t)]
#         perm = self._perm
#         blocks = [perm[offset + np.flatnonzero(r)].tolist() for r in rows]
#         h_alice = np.asarray(await oracle.get_parities(blocks), dtype=np.uint8)
#         dec_i64 = decoded_shuffled.astype(np.int64)
#         h_bob = np.array(
#             [int((r.astype(np.int64) @ dec_i64) & 1) for r in rows],
#             dtype=np.uint8)
#         return np.array_equal(h_alice, h_bob)

#     # ---------------------------------------------------------------- attempt
#     async def _attempt_decode(self, key_chunk, m_new, p, oracle, offset, code,
#                               cached_parities=None):
#         k = len(key_chunk)
#         n = code["n"]; G = code["G"]
#         syst_cols = code["syst_cols"]
#         info_indices = code["info_indices"]
#         parity_cols = code["parity_cols"]
#         perm = self._perm

#         key_arr = np.asarray(key_chunk, dtype=np.int64)
#         llr = np.zeros(n, dtype=np.float32)
#         l_sys = math.log((1.0 - p) / p)
#         llr[syst_cols] = np.where(key_arr[info_indices] == 1, l_sys, -l_sys)

#         n_cached = 0 if cached_parities is None else len(cached_parities)
#         if m_new > 0:
#             new_cols = parity_cols[n_cached:n_cached + m_new]
#             cols = G[:, new_cols]
#             blocks = [perm[np.flatnonzero(cols[:, j]) + offset].tolist()
#                       for j in range(m_new)]
#             new_values = np.asarray(await oracle.get_parities(blocks),
#                                     dtype=np.int64)
#             values = (new_values if cached_parities is None
#                       else np.concatenate([cached_parities, new_values]))
#             sent_idx = parity_cols[:len(values)]
#             llr[sent_idx] = np.where(values == 1, self.LLR_PARITY,
#                                      -self.LLR_PARITY)
#         else:
#             values = np.empty(0, dtype=np.int64)

#         dec_in = tf.convert_to_tensor(llr[None, :], dtype=tf.float32)
#         loop = asyncio.get_running_loop()
#         decoded = await loop.run_in_executor(
#             self._executor,
#             lambda: code["decoder"](dec_in).numpy()[0].astype(np.int64))

#         mism = 0
#         if len(values) > 0:
#             sent_idx = parity_cols[:len(values)]
#             est = (decoded.astype(np.int32) @ G[:, sent_idx].astype(np.int32)) % 2
#             mism = int(np.sum(est != values))
#         return decoded, mism, values

#     # ---------------------------------------------------------------- public
#     async def run(self, key: List[int], qber: float, oracle: ParityOracle):
#         counting = _CountingOracle(oracle)
#         oracle = counting

#         self.bits_revealed = 0
#         self.channel_uses = 0
#         self.round_trips = 0
#         self.stats = {}
#         N = len(key)
#         if N == 0:
#             return list(key), 0, 0, 0

#         perm_list = list(range(N))
#         self.prng.shuffle(perm_list)
#         perm = np.asarray(perm_list, dtype=np.int64)
#         self._perm = perm
#         key_arr_full = np.asarray(key, dtype=np.int64)
#         key_shuffled = key_arr_full[perm].tolist()

#         n_chunks = max(1, math.ceil(N / self.max_k))
#         base = N // n_chunks
#         rem = N % n_chunks
#         sizes = [base + 1] * rem + [base] * (n_chunks - rem)
#         offsets = [0]
#         for s in sizes:
#             offsets.append(offsets[-1] + s)

#         corrected_shuffled = []
#         total_mism = 0
#         clamped = 0
#         failed_indices: List[int] = []
#         f_sw_list = []
#         for c in range(n_chunks):
#             lo = offsets[c]; hi = offsets[c + 1]
#             res, mism, cl, f_sw = await self._run_chunk(
#                 list(key_shuffled[lo:hi]), qber, oracle, offset=lo)
#             corrected_shuffled.extend(res)
#             total_mism += mism
#             if mism > 0:
#                 failed_indices.append(c)
#             clamped += 1 if cl else 0
#             f_sw_list.append((f_sw, hi - lo))

#         corrected = key_arr_full.copy()
#         corrected[perm] = np.asarray(corrected_shuffled, dtype=np.int64)

#         p = min(max(qber, 1e-3), 0.5 - 1e-3)
#         h_p = binary_entropy(p)
#         ideal_est = N * h_p
#         flips = int(np.sum(corrected != key_arr_full))
#         f_emp_denom = (N * binary_entropy(flips / N)) if flips > 0 else 0.0

#         if f_sw_list:
#             total_w = sum(w for _, w in f_sw_list)
#             f_sw_min = sum(f * w for f, w in f_sw_list) / total_w
#         else:
#             f_sw_min = 1.0

#         self.bits_revealed = counting.bits
#         self.channel_uses = counting.calls
#         self.round_trips = counting.calls

#         stop_leak = n_chunks * int(math.ceil(math.log2(self.max_sweep_iter + 1)))

#         f_EC = self.bits_revealed / ideal_est if ideal_est > 0 else float("nan")
#         f_total = (self.bits_revealed + stop_leak) / ideal_est if ideal_est > 0 else float("nan")

#         self.stats = dict(
#             n_bits=N, qber_used=p,
#             leaked_bits=self.bits_revealed,
#             efficiency_f=f_EC,
#             efficiency_f_EC=f_EC,
#             efficiency_f_total=f_total,
#             efficiency_f_emp=(self.bits_revealed / f_emp_denom)
#                              if f_emp_denom > 0 else float("nan"),
#             stop_leak_bits=stop_leak,
#             f_sw_min=f_sw_min,
#             bob_flip_fraction=flips / N,
#             syndrome_mismatches=total_mism,
#             failed_chunks=len(failed_indices),
#             failed_chunk_indices=failed_indices,
#             chunk_boundaries=offsets,
#             clamped_chunks=clamped,
#             chunks=n_chunks,
#             channel_uses=self.channel_uses,
#             round_trips=self.round_trips,
#             target_fer=self.target_fer,
#             hash_bits=self.hash_bits,
#         )
#         self.log(f"Done: leaked={self.bits_revealed}, "
#                  f"f_EC={f_EC:.4f}, f_total={f_total:.4f}, "
#                  f"mism={total_mism}, failed={len(failed_indices)}/{n_chunks}, "
#                  f"clamped={clamped}, oracle_calls={counting.calls}")
#         return corrected.tolist(), self.bits_revealed, total_mism, self.channel_uses

#     # ---------------------------------------------------------------- chunk
#     async def _run_chunk(self, key_chunk, qber, oracle, offset=0):
#         if not SIONNA_AVAILABLE:
#             return list(key_chunk), 1, False, 1.0
#         k = len(key_chunk)
#         p = min(max(qber, 1e-3), 0.5 - 1e-3)
#         try:
#             code = self._get_code(k)
#         except Exception as e:
#             self.log(f"Code setup failed for k={k}: {e}")
#             return list(key_chunk), 1, False, 1.0

#         m_start, f_sw_min, clamped = self._m_start(k, code["n_punctured"], p)
#         m_max = len(code["parity_cols"])

#         if self.fixed_rate is not None:
#             m = min(max(m_start, 1), m_max)
#             try:
#                 decoded, mism, values = await self._attempt_decode(
#                     key_chunk, m, p, oracle, offset, code)
#             except Exception as e:
#                 self.log(f"Decode error at m={m}: {e}")
#                 decoded = np.asarray(key_chunk, dtype=np.int64)
#                 mism = 1
#             if mism == 0:
#                 try:
#                     ok = await self._verify_hash(decoded, offset, k, oracle)
#                 except Exception as e:
#                     self.log(f"Hash error at m={m}: {e}")
#                     ok = False
#                 if not ok:
#                     self.log(f"Hash FAIL at m={m}")
#                     mism = 1
#             status = "OK" if mism == 0 else "FAIL"
#             self.log(f"chunk k={k}: {status} at m={m} "
#                      f"(start={m_start}, max={m_max}, f_sw={f_sw_min:.4f})")
#             return decoded.tolist(), mism, clamped, f_sw_min

#         m = min(max(m_start, 1), m_max)
#         best_decoded = None
#         best_mism = 1
#         iterations = 0
#         cached = None

#         while iterations < self.max_sweep_iter:
#             iterations += 1
#             m_new = m - (0 if cached is None else len(cached))
#             if m_new <= 0:
#                 m = min(m + 1, m_max)
#                 m_new = m - (0 if cached is None else len(cached))
#                 if m_new <= 0:
#                     break
#             try:
#                 decoded, mism, values = await self._attempt_decode(
#                     key_chunk, m_new, p, oracle, offset, code,
#                     cached_parities=cached)
#                 cached = values
#             except Exception as e:
#                 self.log(f"Decode error at m={m}: {e}")
#                 break

#             if mism == 0:
#                 try:
#                     ok = await self._verify_hash(decoded, offset, k, oracle)
#                 except Exception as e:
#                     self.log(f"Hash error at m={m}: {e}")
#                     ok = False
#                 if ok:
#                     best_decoded = decoded
#                     best_mism = 0
#                     break
#                 self.log(f"Hash FAIL at m={m}, continuing sweep")

#             if m >= m_max:
#                 break
#             m_next = min(int(math.ceil(m * self.sweep_factor)), m_max)
#             if m_next == m:
#                 m_next = min(m + 1, m_max)
#             m = m_next

#         status = "OK" if best_mism == 0 else "FAIL"
#         self.log(f"chunk k={k}: {status} "
#                  f"(start={m_start}, max={m_max}, f_sw={f_sw_min:.4f}, "
#                  f"iters={iterations})")

#         corrected_key = (best_decoded.tolist()
#                          if best_decoded is not None else list(key_chunk))
#         return corrected_key, best_mism, clamped, f_sw_min

#     # ---------------------------------------------------------------- offline
#     async def calibrate_m_min(self, key: List[int], qber: float,
#                               oracle: ParityOracle):
#         self._perm = np.arange(len(key), dtype=np.int64)
#         N = len(key)
#         n_chunks = max(1, math.ceil(N / self.max_k))
#         base = N // n_chunks
#         rem = N % n_chunks
#         sizes = [base + 1] * rem + [base] * (n_chunks - rem)
#         offsets = [0]
#         for s in sizes:
#             offsets.append(offsets[-1] + s)

#         out = []
#         for c in range(n_chunks):
#             lo, hi = offsets[c], offsets[c + 1]
#             chunk = list(key[lo:hi])
#             k = len(chunk)
#             code = self._get_code(k)
#             m_max = len(code["parity_cols"])
#             p = min(max(qber, 1e-3), 0.5 - 1e-3)

#             m_min = m_max
#             for m in range(1, m_max + 1):
#                 try:
#                     decoded, mism, _ = await self._attempt_decode(
#                         chunk, m, p, oracle, lo, code, cached_parities=None)
#                 except Exception:
#                     continue
#                 if mism != 0:
#                     continue
#                 ok = await self._verify_hash(decoded, lo, k, oracle)
#                 if ok:
#                     m_min = m
#                     break
#             out.append({"size": k, "m_min": m_min, "m_max": m_max})
#         return out


# # --- self test (NO import from bb84_server_client) -------------------------
# if __name__ == "__main__":
#     async def _main():
#         print("--- Empirical-sweep LDPC test (multi-seed) ---")
#         N = 10000
#         n_seeds = 5

#         class Mock(ParityOracle):
#             def __init__(self, k): self.key = np.asarray(k)
#             async def get_parity(self, idx): return int(self.key[idx].sum() % 2)
#             async def get_parities(self, blocks):
#                 return [int(self.key[b].sum() % 2) for b in blocks]
#             async def apply_privacy_amplification(self, *a, **kw): pass

#         for p in (0.05, 0.08, 0.10, 0.12):
#             f_ec_list, f_tot_list = [], []
#             res_list, leaked_list, fer_list = [], [], []
#             for seed in range(n_seeds):
#                 rng = np.random.default_rng(seed)
#                 alice = rng.integers(0, 2, N).tolist()
#                 bob = [b ^ int(rng.random() < p) for b in alice]
#                 prot = NR_LDPC_Standard_ClientProtocol(
#                     verbose=False, rate=None, seed=seed)
#                 corr, leaked, mism, uses = await prot.run(bob, p, Mock(alice))

#                 s = prot.stats
#                 failed = set(s['failed_chunk_indices'])
#                 bounds = s['chunk_boundaries']
#                 good = []
#                 for c in range(s['chunks']):
#                     if c in failed:
#                         continue
#                     good.extend(range(bounds[c], bounds[c + 1]))
#                 if good:
#                     res_err = sum(alice[i] != corr[i] for i in good)
#                 else:
#                     res_err = -1
#                 res_list.append(res_err if res_err >= 0 else float('nan'))
#                 f_ec_list.append(s['efficiency_f_EC'])
#                 f_tot_list.append(s['efficiency_f_total'])
#                 leaked_list.append(leaked)
#                 fer_list.append(s['failed_chunks'] / s['chunks'])

#             print(f"p={p:.2f}  f_EC={np.nanmean(f_ec_list):.4f}"
#                   f"±{np.nanstd(f_ec_list):.4f}  "
#                   f"f_tot={np.nanmean(f_tot_list):.4f}  "
#                   f"FER={np.mean(fer_list):.3f}  "
#                   f"residual={np.nanmean(res_list):.1f}  "
#                   f"leaked={np.mean(leaked_list):.0f}")

#     if SIONNA_AVAILABLE:
#         asyncio.run(_main())

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

    LLR_PARITY = 20.0

    # def __init__(self, verbose=True, rate=None, seed=None,
    #              num_iter=10, max_k=1000, target_fer=5e-16,
    #              sweep_factor=1.12, max_sweep_iter=10,
    #              hash_bits=512): new1
    def __init__(self, verbose=True, rate=None, seed=None,
                 num_iter=10, max_k=6000, target_fer=5e-16,
                 sweep_factor=1.62, max_sweep_iter=10,
                 hash_bits=512):
    # def __init__(self, verbose=True, rate=None, seed=None,
    #                  num_iter=10, max_k=6000, target_fer=1e-3,
    #                  sweep_factor=1.60, max_sweep_iter=10,
    #                  hash_bits=64): #old1
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


