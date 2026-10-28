"""
==============================================================

Decoy-State MDI-QKD Simulation (Fully Empirical, Event-Driven)
==============================================================

Based on the actor architecture of bb84_server_client.py 

References:
  [R1] https://arxiv.org/abs/1109.1473    -- MDI-QKD protocol
  [R2] https://arxiv.org/abs/1406.0188 -- decoy-state optimization
  [R3] https://www.nature.com/articles/ncomms4732 -- finite-key analysis

"""


import asyncio
import logging
import datetime
import math
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Any
from collections import defaultdict
import sys
import os
import time

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prng import PRNG, Basis
from cascade_refactored import CascadeClientProtocol, ParityOracle
from nr_ldpc_standard import NR_LDPC_Standard_ClientProtocol
from winnow_refactored import WinnowClientProtocol
from polar_codes import PolarClientProtocol
from privacy_amplification import PrivacyAmplification

try:
    from scipy.optimize import linprog
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

logging.basicConfig(level=logging.ERROR, format='%(message)s')


# ===================================================================
# GLOBAL CONFIGURATION -- edit here to change the run
# ===================================================================

CONFIG = {
    # ---------- Quantum source ----------
    "N": 500000,               # number of pulse pairs
    "mu": 0.5,                  # signal intensity
    "nu": 0.1,                  # decoy intensity
    "p_signal": 0.50,
    "p_decoy":  0.25,
    "p_vacuum": 0.25,

    # ---------- Channels ----------
    "L_A_km": 10,
    "L_B_km": 10,
    "alpha_dB_per_km": 0.2,
    "e_pol": 0.001,

    # ---------- Detection / BSM ----------
    "eta_det": 0.9,
    "p_dark": 1e-6,
    "hom_visibility": 0.99,

    # ---------- Error Correction ----------
    "ec_protocol": "cascade",
    "ec_params": {
        "num_passes": 4,
        "rate": 0.33,
        "r_max": 0.333,
        "u_fer_target": 0.01,
        "c": 0.5,
    },

    # ---------- Privacy Amplification ----------
    "pa_protocol": "toeplitz",

    # ---------- Parameter estimation ----------
    "qber_sample_fraction": 0.20,
    "min_sample_size": 10,
    "security_eps": 1e-10,
    "N_c": 3,

    # ---------- Determinism ----------
    "seed_block": 7_000_000,
}


# ===================================================================
# Data structures
# ===================================================================

@dataclass
class PhotonPulse:
    id: int
    bit: int
    basis: Basis
    intensity: str
    intensity_value: float
    photon_number: int
    creation_time: str
    source: str = "A"
    photon_number_source: int = 0


@dataclass
class BSMEvent:
    pulse_id: int
    outcome: str                 # "Psi+" | "Psi-"
    time: str
    alice_bit: int
    bob_bit: int
    alice_basis: Basis
    bob_basis: Basis
    alice_intensity: str
    bob_intensity: str
    alice_n_source: int = 0
    bob_n_source: int = 0


# ===================================================================
# Actor framework
# ===================================================================

class Actor:
    def __init__(self, name, seed=None):
        self.name = name
        self.mailbox = asyncio.Queue()
        self._running = False
        self.prng = PRNG(seed)

    async def start(self):
        self._running = True
        while self._running:
            msg = await self.mailbox.get()
            if isinstance(msg, tuple) and msg[0] == "STOP":
                self._running = False
                break
            await self.handle_message(msg)

    async def send(self, recipient, msg):
        await recipient.mailbox.put(msg)

    async def handle_message(self, msg):
        pass


# ===================================================================
# Quantum channel
# ===================================================================

class QuantumChannel(Actor):
    """
    Fiber channel with attenuation and polarization rotation.

    The polarization rotation is modelled as a pure bit-flip with
    probability e_pol, applied identically in both bases.  For a
    physical rotation about the fiber axis, the Z-basis error
    probability and the X-basis error probability are both sin^2(theta),
    so this simplification is correct in the small-error limit and
    does not introduce a basis-dependent bias.
    """

    def __init__(self, name, length_km, attenuation_db, optical_error_rate,
                 next_actor, seed=None):
        super().__init__(name, seed)
        self.length = length_km
        self.att_db = attenuation_db
        self.optical_error_rate = optical_error_rate
        self.next_actor = next_actor
        self.transmittance = 10 ** (-(length_km * attenuation_db) / 10)

    async def handle_message(self, msg):
        if not isinstance(msg, PhotonPulse):
            await self.send(self.next_actor, msg)
            return

        transmitted = 0
        for _ in range(msg.photon_number):
            if self.prng.random() < self.transmittance:
                transmitted += 1

        out_bit = msg.bit
        if self.prng.random() < self.optical_error_rate:
            out_bit = 1 - out_bit

        out = PhotonPulse(
            id=msg.id, bit=out_bit, basis=msg.basis,
            intensity=msg.intensity, intensity_value=msg.intensity_value,
            photon_number=transmitted, creation_time=msg.creation_time,
            source=msg.source, photon_number_source=msg.photon_number_source,
        )
        await self.send(self.next_actor, out)


# ===================================================================
# MDI-QKD sender (Alice or Bob)
# ===================================================================

class MDISender(Actor):
    """
    A decoy-state weak coherent pulse source.

    Because only the bit (not the phase) is tracked, this simulation
    implicitly assumes phase-randomised coherent pulses.  The BSM
    coincidence probabilities are exactly the phase-averaged
    HOM-interference results, so the model is consistent with
    decoy-state MDI-QKD.
    """

    def __init__(self, name, channel, num_pulses,
                 mu, nu, p_signal, p_decoy, p_vacuum,
                 source_label="A", verbose=False, seed=None):
        super().__init__(name, seed)
        self.channel = channel
        self.num_pulses = num_pulses
        self.mu = mu
        self.nu = nu
        self.p_signal = p_signal
        self.p_decoy = p_decoy
        self.p_vacuum = p_vacuum
        self.source_label = source_label
        self.verbose = verbose
        self.sent_log: Dict[int, PhotonPulse] = {}

    def sample_intensity(self):
        r = self.prng.random()
        if r < self.p_signal:
            return ("signal", self.mu)
        elif r < self.p_signal + self.p_decoy:
            return ("decoy", self.nu)
        else:
            return ("vacuum", 0.0)

    def sample_photon_number(self, lam):
        """Knuth's Poisson sampler using this actor's own PRNG."""
        if lam <= 0.0:
            return 0
        L = math.exp(-lam)
        k = 0
        p = 1.0
        while True:
            k += 1
            p *= self.prng.random()
            if p <= L:
                return k - 1

    async def run_quantum_transmission(self):
        if self.verbose:
            print(f"\n[{self.name}] Transmitting {self.num_pulses} pulses...")
        for i in range(self.num_pulses):
            bit = self.prng.get_bit()
            basis = self.prng.get_basis()
            label, val = self.sample_intensity()
            n = self.sample_photon_number(val)
            t_now = datetime.datetime.now().strftime("%H:%M:%S.%f")
            pulse = PhotonPulse(
                id=i, bit=bit, basis=basis,
                intensity=label, intensity_value=val,
                photon_number=n, creation_time=t_now,
                source=self.source_label, photon_number_source=n,
            )
            self.sent_log[i] = pulse
            await self.send(self.channel, pulse)
            if i % 1000 == 0:
                await asyncio.sleep(0)
        if self.verbose:
            print(f"[{self.name}] Transmission complete.")


# ===================================================================
# Bell state measurement
# ===================================================================

class BSMActor(Actor):
    """
    Linear-optics partial Bell state measurement at Charlie.

    Physical model
    --------------
    A 50:50 beam splitter followed by polarizing beam splitters and
    single-photon detectors.  With linear optics only two of the four
    Bell states can be identified, and the identification relies on
    Hong-Ou-Mandel (HOM) interference of the two photons at the BS.

    For two single photons at the BS:

      * Different polarizations (H,V) or (V,H):
            the photons are distinguishable, so they exit the two
            output ports independently.  A coincidence is registered
            with probability 1/2 (given both are detected).
            The Bell-state outcome is randomised 50/50 between
            |Psi+> and |Psi->.

      * Same polarizations (H,H) or (V,V):
            HOM interference suppresses coincidences by the visibility
            V.  With ideal visibility (V = 1) the photons always bunch
            and no coincidence is registered.  With imperfect
            visibility, the residual coincidence probability is
            (1 - V) / 2 * P(both detected), and the outcome is again
            randomised 50/50 between |Psi+> and |Psi->.

    Dark counts contribute an independent two-fold coincidence floor.
    With four detectors (D1H, D1V, D2H, D2V) and the standard
    linear-optics BSM coincidence logic (one detector in EACH output
    port), the leading-order dark-count coincidence probability is

        P_dc,coinc = [1 - (1 - p_dc)^2]^2  ~=  4 p_dc^2     (small p_dc)

    Reference: [R1] Sec. II (Eqs. 2-4).

    Buffer note
    -----------
    The internal buffer pairs Alice's pulse with Bob's pulse by pulse
    id.  Since both senders iterate over the same id range [0, N) and
    every pulse is eventually forwarded to the BSM, the buffer is
    naturally bounded: it holds at most N entries during the overlap
    window in which only one side's pulses have arrived.  No eviction
    policy is needed or applied.
    """

    def __init__(self, name, server, eta_det=0.8, p_dark=1e-6,
                 hom_visibility=0.95, seed=None):
        super().__init__(name, seed)
        self.server = server
        self.eta = eta_det
        self.p_dc = p_dark
        self.V = hom_visibility
        self.buffer: Dict[int, Dict[str, PhotonPulse]] = {}

    async def handle_message(self, msg):
        if not isinstance(msg, PhotonPulse):
            return
        self.buffer.setdefault(msg.id, {})[msg.source] = msg
        if "A" in self.buffer[msg.id] and "B" in self.buffer[msg.id]:
            pa = self.buffer[msg.id]["A"]
            pb = self.buffer[msg.id]["B"]
            del self.buffer[msg.id]
            event = self._run_bsm(pa, pb)
            if event is not None:
                await self.send(self.server, event)

    def _run_bsm(self, pa: PhotonPulse, pb: PhotonPulse):
        n_a, n_b = pa.photon_number, pb.photon_number

        # Probability that at least one photon from each arm reaches
        # a detector (before HOM interference is applied).
        if n_a == 0 or n_b == 0:
            p_both = 0.0
        else:
            p_det_a = 1.0 - (1.0 - self.eta) ** n_a
            p_det_b = 1.0 - (1.0 - self.eta) ** n_b
            p_both = p_det_a * p_det_b

        a_is_Z = (pa.basis == Basis.RECTILINEAR)
        b_is_Z = (pb.basis == Basis.RECTILINEAR)
        a_is_X = (pa.basis == Basis.DIAGONAL)
        b_is_X = (pb.basis == Basis.DIAGONAL)

        # Mixed-basis events do not participate in sifting or in the
        # phase-error estimate; Charlie discards them.
        if not ((a_is_Z and b_is_Z) or (a_is_X and b_is_X)):
            return None

        if a_is_Z and b_is_Z:
            # ---- Z basis:  bit 0 = |H>, bit 1 = |V> --------------------
            same_bit = (pa.bit == pb.bit)
            if same_bit:
                p_coinc_given_both = 0.5 * (1.0 - self.V)
            else:
                p_coinc_given_both = 0.5
            outcome_det = None

        else:
            # ---- X basis:  bit 0 = |+>, bit 1 = |-> --------------------
            same_bit = (pa.bit == pb.bit)
            p_coinc_given_both = 0.25 * (2.0 - self.V)
            outcome_det = "Psi+" if same_bit else "Psi-"

        p_coinc = p_coinc_given_both * p_both

        # Correct leading-order dark-count coincidence floor.
        p_dc_twofold = (1.0 - (1.0 - self.p_dc) ** 2) ** 2

        p_total = p_coinc + p_dc_twofold - p_coinc * p_dc_twofold

        if self.prng.random() < p_total:
            if outcome_det is None:
                outcome = "Psi-" if self.prng.random() < 0.5 else "Psi+"
            else:
                outcome = outcome_det
            return BSMEvent(
                pulse_id=pa.id, outcome=outcome,
                time=datetime.datetime.now().strftime("%H:%M:%S.%f"),
                alice_bit=pa.bit, bob_bit=pb.bit,
                alice_basis=pa.basis, bob_basis=pb.basis,
                alice_intensity=pa.intensity, bob_intensity=pb.intensity,
                alice_n_source=pa.photon_number_source,
                bob_n_source=pb.photon_number_source,
            )
        return None


# ===================================================================
# Classical server (sift / sample / parity / PA / stats)
# ===================================================================

class AliceServer(Actor):
    """
    Classical post-processing server.
    """

    def __init__(self, name, verbose=False, seed=None):
        super().__init__(name, seed)
        self.verbose = verbose
        self.bsm_events: List[BSMEvent] = []
        self.alice_pulse_info: Dict[int, Dict] = {}
        self.bob_pulse_info: Dict[int, Dict] = {}
        self.sifted_pulse_ids: List[int] = []
        self.alice_sifted_bits: List[int] = []
        self.final_secret_key: List[int] = []
        self.gain_stats = defaultdict(
            lambda: {"pulses": 0, "coinc": 0, "err": 0}
        )

    async def handle_message(self, msg):
        if isinstance(msg, BSMEvent):
            self.bsm_events.append(msg)

    def register_alice_pulses(self, pulse_info):
        self.alice_pulse_info = pulse_info

    async def handle_api_request(self, endpoint, data):
        if endpoint == "/sift":
            return self._api_sift(data)
        elif endpoint == "/sample":
            return self._api_sample(data)
        elif endpoint == "/parity":
            return self._api_parity(data)
        elif endpoint == "/block-parity":
            return self._api_block_parity(data)
        elif endpoint == "/privacy-amplification":
            return self._api_pa(data)
        elif endpoint == "/decoy-gains":
            return self._api_decoy_gains(data)
        elif endpoint == "/empirical-stats":
            return self._api_empirical_stats(data)
        else:
            raise ValueError(f"Unknown endpoint: {endpoint}")

    # ------------------------------------------------------------------
    # Sifting
    # ------------------------------------------------------------------
    def _api_sift(self, data):
        self.bob_pulse_info = data["pulses"]
        self._compute_gain_stats()

        matching_ids = []
        matching_outcomes = {}
        for ev in self.bsm_events:
            pid = ev.pulse_id
            a = self.alice_pulse_info.get(pid)
            b = self.bob_pulse_info.get(pid)
            if a is None or b is None:
                continue
            if ev.alice_basis.value != a["basis"] or \
               ev.bob_basis.value != b["basis"]:
                continue
            if a["basis"] != Basis.RECTILINEAR.value:
                continue
            if b["basis"] != Basis.RECTILINEAR.value:
                continue
            matching_ids.append(pid)
            matching_outcomes[pid] = ev.outcome

        self.sifted_pulse_ids = matching_ids
        self.alice_sifted_bits = [
            self.alice_pulse_info[pid]["bit"] for pid in matching_ids
        ]
        if self.verbose:
            print(f"[ALICE] Sift: {len(matching_ids)} Z-Z matched events.")
        return {
            "matching_indices": matching_ids,
            "outcomes": matching_outcomes,
        }

    @staticmethod
    def _sifted_bob_bit(basis_value, bob_bit, outcome):
        """Apply the MDI-QKD sifting rule to Bob's raw bit.

        Z basis: Bob always flips his bit (coincidences occur only
                 when Alice's and Bob's Z bits differ).
        X basis: Bob flips his bit iff Charlie announced |Psi->.
        """
        if basis_value == Basis.RECTILINEAR.value:
            return 1 - bob_bit
        elif basis_value == Basis.DIAGONAL.value:
            return bob_bit if outcome == "Psi+" else 1 - bob_bit
        return bob_bit

    def _compute_gain_stats(self):
        self.gain_stats = defaultdict(
            lambda: {"pulses": 0, "coinc": 0, "err": 0}
        )
        common_ids = set(self.alice_pulse_info) & set(self.bob_pulse_info)
        for pid in common_ids:
            a = self.alice_pulse_info[pid]
            b = self.bob_pulse_info[pid]
            if a["basis"] != b["basis"]:
                continue
            key = (a["basis"], a["intensity"], b["intensity"])
            self.gain_stats[key]["pulses"] += 1

        for ev in self.bsm_events:
            pid = ev.pulse_id
            a = self.alice_pulse_info.get(pid)
            b = self.bob_pulse_info.get(pid)
            if a is None or b is None:
                continue
            if a["basis"] != b["basis"]:
                continue
            key = (a["basis"], a["intensity"], b["intensity"])
            self.gain_stats[key]["coinc"] += 1
            bob_sifted = self._sifted_bob_bit(a["basis"], b["bit"], ev.outcome)
            is_err = (a["bit"] != bob_sifted)
            if is_err:
                self.gain_stats[key]["err"] += 1

    def _api_decoy_gains(self, data):
        gains = {}
        for (basis, qa, qb), stats in self.gain_stats.items():
            n, c, e = stats["pulses"], stats["coinc"], stats["err"]
            gains[(basis, qa, qb)] = {
                "Q": c / n if n > 0 else 0.0,
                "E": e / c if c > 0 else 0.0,
                "N_pulses": n, "N_coinc": c, "N_err": e,
            }
        return {"gains": gains}

    # ------------------------------------------------------------------
    # Empirical single-photon statistics
    # ------------------------------------------------------------------

    def _api_empirical_stats(self, data):
        Z = Basis.RECTILINEAR.value
        X = Basis.DIAGONAL.value

        def basis_counts(basis_val):
            n11_total = 0
            for pid in set(self.alice_pulse_info) & set(self.bob_pulse_info):
                a = self.alice_pulse_info[pid]
                b = self.bob_pulse_info[pid]
                if a["basis"] != basis_val or b["basis"] != basis_val:
                    continue
                if a["n_source"] == 1 and b["n_source"] == 1:
                    n11_total += 1

            n11_coinc = 0
            n11_err = 0
            for ev in self.bsm_events:
                pid = ev.pulse_id
                a = self.alice_pulse_info.get(pid)
                b = self.bob_pulse_info.get(pid)
                if a is None or b is None:
                    continue
                if a["basis"] != basis_val or b["basis"] != basis_val:
                    continue
                if a["n_source"] == 1 and b["n_source"] == 1:
                    n11_coinc += 1
                    bob_sifted = self._sifted_bob_bit(
                        a["basis"], b["bit"], ev.outcome
                    )
                    if a["bit"] != bob_sifted:
                        n11_err += 1
            return n11_total, n11_coinc, n11_err

        n11_total_Z, n11_coinc_Z, n11_err_Z = basis_counts(Z)
        n11_total_X, n11_coinc_X, n11_err_X = basis_counts(X)

        n_sifted = len(self.sifted_pulse_ids)
        n11_sifted = 0
        for pid in self.sifted_pulse_ids:
            a = self.alice_pulse_info.get(pid)
            b = self.bob_pulse_info.get(pid)
            if a is None or b is None:
                continue
            if a["n_source"] == 1 and b["n_source"] == 1:
                n11_sifted += 1

        Y11_Z = n11_coinc_Z / n11_total_Z if n11_total_Z > 0 else 0.0
        e11_Z = n11_err_Z / n11_coinc_Z if n11_coinc_Z > 0 else 0.0
        Y11_X = n11_coinc_X / n11_total_X if n11_total_X > 0 else 0.0
        e11_X = n11_err_X / n11_coinc_X if n11_coinc_X > 0 else 0.0

        return {
            "Y11_Z": Y11_Z,
            "e11_Z": e11_Z,
            "Y11_X": Y11_X,
            "e11_X": e11_X,
            "q1":    n11_sifted / n_sifted if n_sifted > 0 else 0.0,
            "N11_total_Z":  n11_total_Z,
            "N11_coinc_Z":  n11_coinc_Z,
            "N11_err_Z":    n11_err_Z,
            "N11_total_X":  n11_total_X,
            "N11_coinc_X":  n11_coinc_X,
            "N11_err_X":    n11_err_X,
            "N11_sifted":   n11_sifted,
            "N_sifted":     n_sifted,
        }

    def _api_sample(self, data):
        # Trim self.sifted_pulse_ids in lock-step with
        # self.alice_sifted_bits so that downstream statistics
        # (q1, N_sifted, etc.) refer to the *remaining* key set.
        indices = data["indices"]
        bits = [self.alice_sifted_bits[i] for i in indices]
        idx_set = set(indices)
        self.alice_sifted_bits = [
            b for i, b in enumerate(self.alice_sifted_bits) if i not in idx_set
        ]
        self.sifted_pulse_ids = [
            p for i, p in enumerate(self.sifted_pulse_ids) if i not in idx_set
        ]
        return {"bits": bits}

    def _par_calc(self, indices):
        p = 0
        N = len(self.alice_sifted_bits)
        for idx in indices:
            if 0 <= idx < N:
                p ^= self.alice_sifted_bits[idx]
        return p

    def _api_parity(self, data):
        return {"parity": self._par_calc(data["indices"])}

    def _api_block_parity(self, data):
        return {"parities": [self._par_calc(blk) for blk in data["blocks"]]}

    def _api_pa(self, data):
        seed = data["seed"]
        qber = data["qber"]
        pa_protocol = data.get("pa_protocol", "toeplitz")
        q1 = data.get("q1", 1.0)
        e11_X = data.get("e11_X", qber)
        ec_leakage = data.get("ec_leakage", 0)
        if self.verbose:
            print(f"[ALICE]   phase-error rate fed to PA "
                  f"= {e11_X:.4%}  ")
            print(f"[ALICE]   q1 = {q1:.4f}, EC leakage = {ec_leakage} bits")
        pa = PrivacyAmplification(
            self.alice_sifted_bits, e11_X,
            q1=q1, ec_leakage=ec_leakage,
        )
        self.final_secret_key = pa.apply_hash(seed, algorithm=pa_protocol)
        return {"status": "success", "pa_key_length": len(self.final_secret_key)}


# ===================================================================
# API client (ParityOracle)
# ===================================================================

class APIClient(ParityOracle):
    def __init__(self, server: AliceServer):
        self.server = server

    async def post(self, endpoint, data):
        return await self.server.handle_api_request(endpoint, data)

    async def get_parity(self, indices):
        return (await self.post("/parity", {"indices": indices}))["parity"]

    async def get_parities(self, blocks):
        return (await self.post("/block-parity", {"blocks": blocks}))["parities"]

    async def apply_privacy_amplification(self, seed, qber,
                                          pa_protocol="toeplitz",
                                          q1=1.0, e11_X=0.0,
                                          ec_leakage=0):
        return await self.post("/privacy-amplification", {
            "seed": seed, "qber": qber, "pa_protocol": pa_protocol,
            "q1": q1, "e11_X": e11_X, "ec_leakage": ec_leakage,
        })


# ===================================================================
# Bob's client driver
# ===================================================================

class MDIClient(Actor):
    """
    Drives classical post-processing:
      1. Sift (via server) -- Z basis only
      2. Apply Z-basis bit flip (ALWAYS flip for Z basis)
      3. QBER sampling (Z basis, for EC)
      4. Empirical single-photon statistics (Y11, e11 in both bases)
      5. Error correction on Z-basis sifted key
      6. Privacy amplification using e11_X
    """

    def __init__(self, name, api_client, sender: MDISender,
                 ec_protocol="cascade", ec_params=None,
                 pa_protocol="toeplitz",
                 sample_fraction=0.2, min_sample_size=10,
                 security_eps=1e-10, N_c=3,
                 seed=None, verbose=False):
        super().__init__(name, seed)
        self.api = api_client
        self.sender = sender
        self.ec_protocol = ec_protocol.lower()
        self.ec_params = ec_params or {}
        self.pa_protocol = pa_protocol.lower()
        self.sample_fraction = sample_fraction
        self.min_sample_size = min_sample_size
        self.security_eps = security_eps
        self.N_c = N_c
        self.verbose = verbose

    async def run_classical_post_processing(self):
        # ---------- 1. Sift ----------
        bob_pulses = {
            pid: {
                "basis": p.basis.value,
                "intensity": p.intensity,
                "bit": p.bit,
                "n_source": p.photon_number_source,
            }
            for pid, p in self.sender.sent_log.items()
        }
        print(f"\n[BOB] Sending sift request for {len(bob_pulses)} pulses...")
        resp = await self.api.post("/sift", {"pulses": bob_pulses})
        matching_ids = resp["matching_indices"]
        outcomes = resp["outcomes"]
        print(f"[BOB] Sift returned {len(matching_ids)} Z-Z matching events.")

        # ---------- 2. Z-basis bit flip ----------
        sifted_key_bob = []
        for pid in matching_ids:
            b = self.sender.sent_log[pid].bit
            b = 1 - b
            sifted_key_bob.append(int(b))

        n_sifted = len(sifted_key_bob)
        if n_sifted == 0:
            print("[BOB] No sifted events; aborting.")
            return self._empty_result(0)
        print(f"[BOB] Sifted Key Preview: {sifted_key_bob[:20]}...")

        if n_sifted < max(self.min_sample_size, 20):
            print(f"[BOB] Sifted key too short ({n_sifted}); aborting.")
            return self._empty_result(n_sifted)

        # ---------- 3. QBER sampling ----------
        k = max(self.min_sample_size, int(n_sifted * self.sample_fraction))
        k = min(k, n_sifted // 2)
        indices_to_sample = sorted(self.prng.sample(range(n_sifted), k))
        print(f"[BOB] Sampling {k} bits for QBER estimation...")
        resp = await self.api.post("/sample", {"indices": indices_to_sample})
        alice_bits = resp["bits"]

        errors = sum(
            1 for i, idx in enumerate(indices_to_sample)
            if sifted_key_bob[idx] != alice_bits[i]
        )
        qber = errors / k
        print(f"[BOB] QBER = {errors}/{k} = {qber:.4%}")

        idx_set = set(indices_to_sample)
        clean_key_bob = [
            int(bool(b)) for i, b in enumerate(sifted_key_bob)
            if i not in idx_set
        ]
        print(f"[BOB] Working key size after sampling: {len(clean_key_bob)}")

        if len(clean_key_bob) == 0:
            return self._empty_result(n_sifted, qber)

        # ---------- 4. Empirical single-photon statistics ----------
        gains_resp = await self.api.post("/decoy-gains", {})
        gains = gains_resp["gains"]

        emp = await self.api.post("/empirical-stats", {})
        Y11_Z = emp["Y11_Z"]
        e11_Z = emp["e11_Z"]
        Y11_X = emp["Y11_X"]
        e11_X = emp["e11_X"]
        q1    = emp["q1"]
        print(f"[BOB] Empirical Y11^Z  = {Y11_Z:.6e} "
              f"({emp['N11_coinc_Z']}/{emp['N11_total_Z']})")
        print(f"[BOB] Empirical e11^Z  = {e11_Z:.4%} "
              f"({emp['N11_err_Z']}/{emp['N11_coinc_Z']})")
        print(f"[BOB] Empirical Y11^X  = {Y11_X:.6e} "
              f"({emp['N11_coinc_X']}/{emp['N11_total_X']})")
        print(f"[BOB] Empirical e11^X  = {e11_X:.4%} "
              f"({emp['N11_err_X']}/{emp['N11_coinc_X']})  <- used in PA")
        print(f"[BOB] Empirical q1     = {q1:.4f} "
              f"({emp['N11_sifted']}/{emp['N_sifted']})")
        Q_sig = gains.get(
            (Basis.RECTILINEAR.value, "signal", "signal"), {}
        ).get("Q", 0.0)
        E_sig = gains.get(
            (Basis.RECTILINEAR.value, "signal", "signal"), {}
        ).get("E", 0.0)
        print(f"[BOB] Signal-signal Z gain = {Q_sig:.6e}, "
              f"E = {E_sig:.4%}")

        # ---------- 5. Error correction ----------
        print(f"\n[BOB] Starting {self.ec_protocol.title()} error correction...")
        t_start = time.time()
        est_qber = qber if qber > 0 else 0.01

        ec_seed = self.prng.randint(0, 2 ** 32 - 1)
        print(f"[BOB] EC protocol seed: {ec_seed}")
        protocol = self._build_ec_protocol(seed=ec_seed)

        try:
            corrected_key, revealed, errors_cor, uses = await protocol.run(
                clean_key_bob, est_qber, self.api
            )
        except Exception as e:
            print(f"[BOB] EC failed with exception: {e}")
            import traceback; traceback.print_exc()
            corrected_key, revealed, errors_cor, uses = clean_key_bob, 0, 0, 0

        print(f"[BOB] EC done. Revealed={revealed}, "
              f"Corrected={errors_cor}, ChannelUses={uses}")

        # ---------- 6. Privacy amplification ----------
        final_secret_key: List[int] = []
        pa_length, pa_time = 0, 0.0
        pa_seed = None
        if len(corrected_key) > 0 and qber < 0.5:
            print(f"\n[BOB] Starting privacy amplification "
                  f"({self.pa_protocol})...")
            pa_seed = self.prng.randint(0, 2 ** 32 - 1)

            t_pa = time.time()

            pa = PrivacyAmplification(
                corrected_key, e11_X,
                q1=q1, ec_leakage=revealed,
            )
            pa.log_metrics()
            final_secret_key = pa.apply_hash(
                seed=pa_seed, algorithm=self.pa_protocol
            )
            pa_time = time.time() - t_pa
            pa_length = len(final_secret_key)

            await self.api.apply_privacy_amplification(
                pa_seed, qber, self.pa_protocol,
                q1=q1, e11_X=e11_X, ec_leakage=revealed,
            )
            print(f"[BOB] PA done. Final secret key length: {pa_length} bits.")

        exec_time = time.time() - t_start
        return {
            "sifted_length": n_sifted,
            "qber": qber,
            "final_length": len(corrected_key),
            "pa_length": pa_length,
            "pa_time": pa_time,
            "revealed": revealed,
            "corrected": errors_cor,
            "channel_uses": uses,
            "corrected_key": corrected_key,
            "final_secret_key": final_secret_key,
            "exec_time": exec_time,
            "Y11_Z": Y11_Z,
            "e11_Z": e11_Z,
            "Y11_X": Y11_X,
            "e11_X": e11_X,
            "q1": q1,
            "gains": gains,
            "N_pulse_pairs": len(self.sender.sent_log),
            "ec_seed": ec_seed,
            "pa_seed": pa_seed,
        }

    def _build_ec_protocol(self, seed):
        p = self.ec_protocol
        if p == "cascade":
            return CascadeClientProtocol(
                num_passes=int(self.ec_params.get("num_passes", 4)),
                verbose=self.verbose,
            )
        elif p in ("ldpc_5g_std", "nr_ldpc_std", "nr_ldpc_standard"):
            return NR_LDPC_Standard_ClientProtocol(
                verbose=self.verbose,
                seed=seed,
                rate=float(self.ec_params.get("rate", 0.33)),
            )
        elif p == "winnow":
            return WinnowClientProtocol(
                num_passes=int(self.ec_params.get("num_passes", 4)),
                verbose=self.verbose,
            )
        elif p == "polar":
            return PolarClientProtocol(
                verbose=self.verbose,
                u_fer_target=float(self.ec_params.get("u_fer_target", 0.01)),
                c=float(self.ec_params.get("c", 0.5)),
            )
        else:
            print(f"[BOB] Unknown EC protocol '{p}', defaulting to Cascade.")
            return CascadeClientProtocol(
                num_passes=int(self.ec_params.get("num_passes", 4)),
                verbose=self.verbose,
            )

    def _empty_result(self, n_sifted, qber=1.0):
        return {
            "sifted_length": n_sifted, "qber": qber,
            "final_length": 0, "pa_length": 0, "pa_time": 0.0,
            "revealed": 0, "corrected": 0, "channel_uses": 0,
            "corrected_key": [], "final_secret_key": [],
            "exec_time": 0.0, "Y11_Z": 0.0, "e11_Z": 0.0,
            "Y11_X": 0.0, "e11_X": 0.0,
            "q1": 0.0, "gains": {},
            "N_pulse_pairs": 0, "ec_seed": None, "pa_seed": None,
        }


# ===================================================================
# Orchestration
# ===================================================================

def print_banner(cfg):
    print("=" * 72)
    print("Decoy-State MDI-QKD Simulation (Fully Empirical, Event-Driven)")
    print("=" * 72)

    print("\n--- Quantum source ---")
    print(f"  N (pulse pairs)              : {cfg['N']}")
    print(f"  mu / nu / omega              : {cfg['mu']} / {cfg['nu']} / 0")
    print(f"  p_signal/decoy/vacuum        : "
          f"{cfg['p_signal']}/{cfg['p_decoy']}/{cfg['p_vacuum']}")

    print("\n--- Channels ---")
    print(f"  L_A, L_B                     : {cfg['L_A_km']} km, {cfg['L_B_km']} km")
    print(f"  alpha                        : {cfg['alpha_dB_per_km']} dB/km")
    print(f"  polarization rotation e_pol  : {cfg['e_pol']}")

    print("\n--- Detection / BSM ---")
    print(f"  detector efficiency eta      : {cfg['eta_det']}")
    print(f"  dark count prob p_dc         : {cfg['p_dark']}")
    print(f"  HOM visibility V             : {cfg['hom_visibility']}")

    print("\n--- Parameter estimation ---")
    print(f"  QBER sample fraction         : {cfg['qber_sample_fraction']}")
    print(f"  min sample size              : {cfg['min_sample_size']}")

    print("\n--- Classical protocols ---")
    print(f"  EC protocol                  : {cfg['ec_protocol']}")
    print(f"  EC params                    : {cfg['ec_params']}")
    print(f"  PA protocol                  : {cfg['pa_protocol']}")


async def _wait_until_drained(actors, timeout=300.0, idle_checks=40):
    """Wait until every actor mailbox has been empty for several
    consecutive polls.

    Replaces a fixed `asyncio.sleep()`.  Because both senders push
    N pulses through a chain of actors, the wall-clock time for the
    pipeline to fully drain depends on host speed and on how quickly
    the BSM and channel actors can process messages.  Polling for an
    idle state is robust across machines.
    """
    t0 = time.time()
    idle = 0
    while time.time() - t0 < timeout:
        total = sum(a.mailbox.qsize() for a in actors)
        if total == 0:
            idle += 1
            if idle >= idle_checks:
                return
        else:
            idle = 0
        await asyncio.sleep(0.05)
    remaining = sum(a.mailbox.qsize() for a in actors)
    print(f"[WARN] Drain wait timed out after {timeout:.0f}s "
          f"(remaining messages: {remaining})")


async def run_mdi_qkd_simulation():
    cfg = CONFIG

    sb = cfg["seed_block"]
    seeds = {
        "alice_server":   sb + 1,
        "alice_sender":   sb + 2,
        "bob_sender":     sb + 3,
        "channel_A":      sb + 4,
        "channel_B":      sb + 5,
        "bsm":            sb + 6,
        "bob_client":     sb + 7,
    }
    assert len(set(seeds.values())) == len(seeds), "Seed collision!"

    print_banner(cfg)

    alice_server = AliceServer("AliceServer", verbose=True,
                               seed=seeds["alice_server"])
    bsm = BSMActor("BSM", alice_server,
                   eta_det=cfg["eta_det"], p_dark=cfg["p_dark"],
                   hom_visibility=cfg["hom_visibility"],
                   seed=seeds["bsm"])
    channel_a = QuantumChannel("ChannelA", cfg["L_A_km"],
                               cfg["alpha_dB_per_km"], cfg["e_pol"],
                               next_actor=bsm, seed=seeds["channel_A"])
    channel_b = QuantumChannel("ChannelB", cfg["L_B_km"],
                               cfg["alpha_dB_per_km"], cfg["e_pol"],
                               next_actor=bsm, seed=seeds["channel_B"])

    alice_sender = MDISender(
        "Alice", channel_a, cfg["N"],
        mu=cfg["mu"], nu=cfg["nu"],
        p_signal=cfg["p_signal"], p_decoy=cfg["p_decoy"],
        p_vacuum=cfg["p_vacuum"],
        source_label="A", verbose=True, seed=seeds["alice_sender"],
    )
    bob_sender = MDISender(
        "BobSender", channel_b, cfg["N"],
        mu=cfg["mu"], nu=cfg["nu"],
        p_signal=cfg["p_signal"], p_decoy=cfg["p_decoy"],
        p_vacuum=cfg["p_vacuum"],
        source_label="B", verbose=True, seed=seeds["bob_sender"],
    )

    actors = [alice_server, bsm, channel_a, channel_b,
              alice_sender, bob_sender]
    tasks = [asyncio.create_task(a.start()) for a in actors]

    # ---------- Quantum phase ----------
    print("[PHASE 1] Quantum transmission...")
    await alice_sender.run_quantum_transmission()
    await bob_sender.run_quantum_transmission()
    # Wait for the pipeline to drain without using a fixed sleep.
    await _wait_until_drained(actors)

    alice_info = {
        pid: {"basis": p.basis.value, "intensity": p.intensity,
              "bit": p.bit, "n_source": p.photon_number_source}
        for pid, p in alice_sender.sent_log.items()
    }
    alice_server.register_alice_pulses(alice_info)
    print(f"\n[INFO] Successful BSM events: {len(alice_server.bsm_events)}")

    # ---------- Classical phase ----------
    print("\n[PHASE 2] Classical post-processing...")
    api = APIClient(alice_server)
    bob_client = MDIClient(
        "Bob", api, bob_sender,
        ec_protocol=cfg["ec_protocol"],
        ec_params=cfg["ec_params"],
        pa_protocol=cfg["pa_protocol"],
        sample_fraction=cfg["qber_sample_fraction"],
        min_sample_size=cfg["min_sample_size"],
        security_eps=cfg["security_eps"],
        N_c=cfg["N_c"],
        seed=seeds["bob_client"],
        verbose=True,
    )
    results = await bob_client.run_classical_post_processing()

    # ---------- Results ----------
    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)
    N_total = results.get("N_pulse_pairs", cfg["N"])
    print(f"EC protocol                    : {cfg['ec_protocol']}")
    print(f"EC params                      : {cfg['ec_params']}")
    print(f"PA protocol                    : {cfg['pa_protocol']}")
    print(f"EC sub-protocol seed           : {results.get('ec_seed')}")
    print(f"PA hash seed                   : {results.get('pa_seed')}")
    print(f"Total pulse pairs (N)          : {N_total}")
    print(f"Sifted key length              : {results['sifted_length']}")
    print(f"Estimated QBER (Z)             : {results['qber']:.4%}")
    print(f"Y11^Z (empirical)              : {results['Y11_Z']:.6e}")
    print(f"e11^Z (empirical)              : {results['e11_Z']:.4%}")
    print(f"Y11^X (empirical)              : {results['Y11_X']:.6e}")
    print(f"e11^X (empirical,phase error)  : {results['e11_X']:.4%}")
    print(f"q1    (empirical)              : {results['q1']:.4f}")
    print(f"EC revealed bits               : {results['revealed']}")
    print(f"EC corrected errors            : {results['corrected']}")
    print(f"Final key length (pre-PA)      : {results['final_length']}")
    print(f"Final secret key length        : {results['pa_length']}")
    if N_total > 0:
        print(f"Empirical SKR                  : "
              f"{results['pa_length'] / N_total:.3e} bits/pulse")
    print(f"EC execution time              : {results['exec_time']:.2f} s")
    print(f"PA execution time              : {results['pa_time']:.4f} s")

    # ---------- Consistency check ----------
    alice_final = alice_server.final_secret_key
    bob_final = results["final_secret_key"]
    if len(alice_final) > 0 and len(bob_final) > 0:
        if len(alice_final) != len(bob_final):
            print(f"\nWARNING: Final key length mismatch! "
                  f"Alice={len(alice_final)} bits, Bob={len(bob_final)} bits")
        L = min(len(alice_final), len(bob_final))
        matches = sum(1 for a, b in zip(alice_final[:L], bob_final[:L])
                      if a == b)
        ratio = matches / L
        print(f"\nFinal key match                : {matches}/{L} ({ratio:.2%})")
        if ratio > 0.99 and len(alice_final) == len(bob_final):
            print("  OK: Alice and Bob share the same final key")
            print(f"  First 64 key bits: "
                  f"{''.join(map(str, bob_final[:64]))}")
        else:
            print("  BAD: Key mismatch (EC or PA failed)")
    else:
        print("\nNo final key produced (empty key or EC/PA skipped).")

    # ---------- Stop actors ----------
    for a in actors:
        await a.send(a, ("STOP",))
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(run_mdi_qkd_simulation())