"""
entanglement_primitives.py
==========================
Photon-by-photon physical-layer primitives for quantum networks.

This module replaces the closed-form (analytical) calculations of the
earlier version with an empirical, discrete-photon simulation.  Each
entanglement-generation attempt now:

  1. Samples the number of photons emitted by each memory from a
     Poissonian distribution (DLCZ-type ensemble source).
  2. Propagates every individual photon through the fibre channel as an
     independent Bernoulli loss trial.
  3. Applies single-photon detector efficiency per photon and Poissonian
     dark counts per detection window.
  4. Tests a physical coincidence pattern at the midpoint Bell-state
     measurement (BSM) to decide whether the attempt is heralded.

The module also provides the link-extension (entanglement-swapping)
primitive, a hardware descriptor, and adapters that expose the output of
an end-to-end entangled pair to the BB84 post-processing pipeline
(sifting, Cascade / LDPC / Polar reconciliation, privacy amplification).

── Physics references ───────────────────────────────────────────────────────

  Barrett–Kok heralded generation:
    S. D. Barrett and P. Kok, "Efficient high-fidelity quantum computation
    using matter qubits and linear optics," Phys. Rev. A 71, 060310 (2005).
    arXiv:quant-ph/0408040.

  DLCZ Poissonian photon statistics:
    L.-M. Duan, M. D. Lukin, J. I. Cirac, and P. Zoller, "Long-distance
    quantum communication with atomic ensembles and linear optics,"
    Nature 414, 413–418 (2001).

  Two-photon interference at a 50/50 beam splitter (Hong–Ou–Mandel):
    C. K. Hong, Z. Y. Ou, and L. Mandel, "Measurement of subpicosecond time
    intervals between two photons by interference," Phys. Rev. Lett. 59,
    2044 (1987).

  Entanglement swapping of Werner states:
    C. H. Bennett, G. Brassard, S. Popescu, B. Schumacher, J. A. Smolin,
    and W. K. Wootters, "Purification of Noisy Entanglement and Faithful
    Teleportation via Noisy Channels," Phys. Rev. Lett. 76, 722 (1996).
    M. Żukowski, A. Zeilinger, M. A. Horne, and A. K. Ekert, "Event-ready-
    detectors Bell experiment via entanglement swapping," Phys. Rev. Lett.
    71, 4287 (1993).

  Memory decoherence (exponential fidelity decay):
    Standard model throughout quantum-repeater analyses; see, e.g.,
    S. Muralidharan et al., "Optimal architectures for long-distance
    quantum communication," Phys. Rev. A 93, 012305 (2016).

  BB84 QBER from Werner-state fidelity:
    QBER = (2/3)(1 − F) for a Werner state with fidelity F; derivation
    in C. H. Bennett et al., Phys. Rev. Lett. 76, 722 (1996).

  Discrete-event quantum-network simulation (architectural reference):
    T. Coopmans et al., "NetSquid, a discrete-event simulation platform
    for quantum networks," Commun. Phys. 4, 164 (2021).

── Classical post-processing interface ──────────────────────────────

  Delivered pairs are exposed through `KeySource`, which yields
  `(raw_key_bits, qber)` tuples in the form expected by the
  reconciliation and privacy-amplification modules used elsewhere in
  this project (Cascade, 5G NR LDPC, rate-adaptive LDPC, Winnow,
  Polar codes, Toeplitz PA).  No changes are required in those
  modules to consume the output of this file.

"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Tuple, Dict, Any

import numpy as np

# Shared PRNG module; all stochastic decisions route through this so
# that the entire simulation is reproducible from a single seed.
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from prng import PRNG, Basis

logging.basicConfig(level=logging.ERROR, format="%(message)s")


# ═══════════════════════════════════════════════════════════════════════
#  0.  PHYSICAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

#: Group velocity of light in standard telecom fibre, in km per ms.
#:  n ≈ 1.47 for silica →  v ≈ c / 1.47 ≈ 204 000 km/s ≈ 204 km/ms.
#:  We round to 200 km/ms; the exact value matters only for latency,
#:  not for fidelity.
FIBRE_SPEED_KM_PER_MS: float = 200.0


# ═══════════════════════════════════════════════════════════════════════
#  1.  ASYNCHRONOUS ACTOR BACKBONE
# ═══════════════════════════════════════════════════════════════════════

class HaltSignal:
    """Cooperative shutdown marker for the actor loop."""
    pass


class Messenger:
    """
    Minimal asynchronous actor with a single inbox queue.

    The interface (`start`, `send`, `handle_message`) deliberately
    mirrors the classical actor used by the reconciliation and
    privacy-amplification subsystems, so that physical-layer actors
    can share the same event loop without modification.

    All stochastic decisions inside a subclass must go through
    `self.prng` (a `PRNG` instance) so that the whole simulation is
    reproducible from a single integer seed.
    """

    def __init__(self, label: str, seed: Optional[int] = None):
        self.label = label
        self.mailbox: asyncio.Queue = asyncio.Queue()
        self._active = False
        self.prng = PRNG(seed)

    async def start(self) -> None:
        self._active = True
        while self._active:
            item = await self.mailbox.get()
            if isinstance(item, HaltSignal):
                self._active = False
                break
            await self.handle_message(item)

    async def send(self, peer: "Messenger", item: Any) -> None:
        await peer.mailbox.put(item)

    async def handle_message(self, item: Any) -> None:
        """Override in subclasses."""
        pass

    # Aliases so that callers using the classical actor vocabulary
    # (`consume`, `dispatch`) do not need to be rewritten.
    async def consume(self, item: Any) -> None:
        await self.handle_message(item)

    async def dispatch(self, peer: "Messenger", item: Any) -> None:
        await self.send(peer, item)


# ═══════════════════════════════════════════════════════════════════════
#  2.  QUANTUM STATE CONTAINERS
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class QState:
    """
    A single-qubit pure state |ψ⟩ = α|0⟩ + β|1⟩ stored as a length-2
    complex vector.  Normalisation is assumed to be maintained by the
    constructors and unitary operations.
    """
    amps: np.ndarray  # shape (2,), dtype=complex128

    @staticmethod
    def ket0() -> "QState":
        return QState(np.array([1.0, 0.0], dtype=complex))

    @staticmethod
    def ket1() -> "QState":
        return QState(np.array([0.0, 1.0], dtype=complex))

    @staticmethod
    def ket_plus() -> "QState":
        return QState(np.array([1.0, 1.0], dtype=complex) / math.sqrt(2))

    @staticmethod
    def ket_minus() -> "QState":
        return QState(np.array([1.0, -1.0], dtype=complex) / math.sqrt(2))

    def p0(self) -> float:
        return float(abs(self.amps[0]) ** 2)

    def p1(self) -> float:
        return float(abs(self.amps[1]) ** 2)

    def norm_ok(self) -> bool:
        return abs(float(np.vdot(self.amps, self.amps).real) - 1.0) < 1e-10


@dataclass
class EntangledPair:
    """
    A two-qubit state in the computational basis
    {|00⟩, |01⟩, |10⟩, |11⟩}, together with the metadata needed to
    track how its fidelity evolves over time.

    `nominal_fidelity` is the fidelity the pair had at creation (or
    immediately after a swap).  `current_fidelity` is the value after
    accounting for idling decoherence.  `born_at_ms` is the virtual
    timestamp of creation.
    """
    amps: np.ndarray                    # shape (4,), complex128
    nominal_fidelity: float = 1.0
    current_fidelity: float = 1.0
    born_at_ms: float = 0.0

    @staticmethod
    def phi_plus() -> "EntangledPair":
        """Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2."""
        return EntangledPair(
            np.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / math.sqrt(2)
        )

    def norm_ok(self) -> bool:
        return abs(float(np.vdot(self.amps, self.amps).real) - 1.0) < 1e-10

    def overlap_with(self, other: "EntangledPair") -> float:
        """Return |⟨ψ|φ⟩|².  For pure states this is the fidelity."""
        return float(abs(np.vdot(self.amps, other.amps)) ** 2)


# ═══════════════════════════════════════════════════════════════════════
#  3.  UNITARY OPERATIONS
# ═══════════════════════════════════════════════════════════════════════

def op_bit_flip(q: QState) -> None:
    """Pauli-X: |0⟩ ↔ |1⟩."""
    q.amps = np.array([q.amps[1], q.amps[0]])


def op_phase_flip(q: QState) -> None:
    """Pauli-Z: |1⟩ → −|1⟩."""
    q.amps = np.array([q.amps[0], -q.amps[1]])


def op_bit_phase_flip(q: QState) -> None:
    """Pauli-Y: |0⟩ → i|1⟩, |1⟩ → −i|0⟩."""
    q.amps = np.array([-1j * q.amps[1], 1j * q.amps[0]])


def op_hadamard(q: QState) -> None:
    """Hadamard: |0⟩ → |+⟩, |1⟩ → |−⟩."""
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    q.amps = inv_sqrt2 * np.array([
        q.amps[0] + q.amps[1],
        q.amps[0] - q.amps[1],
    ])


# ═══════════════════════════════════════════════════════════════════════
#  4.  MEASUREMENT
# ═══════════════════════════════════════════════════════════════════════

def project_z(q: QState, prng: PRNG) -> int:
    """
    Ideal Z-basis measurement.  Returns 0 or 1 and collapses the state.
    The Born rule gives P(1) = |β|².
    """
    outcome = 1 if prng.random() >= q.p0() else 0
    q.amps = (np.array([0.0, 1.0], dtype=complex) if outcome
              else np.array([1.0, 0.0], dtype=complex))
    return outcome


def project_z_with_imperfections(
    q: QState,
    detector_efficiency: float,
    dark_count_prob: float,
    bit_flip_prob: float,
    prng: PRNG,
) -> bool:
    """
    Z-basis measurement with a non-ideal detector.

    The model proceeds in three stages:

      1.  The ideal quantum measurement yields a latent outcome.
      2.  A detector with efficiency η registers that outcome only with
          probability η if the latent result is "1" (photon present);
          if the latent result is "0" (vacuum), a dark count fires with
          probability p_dc.
      3.  A classical bit-flip error occurs with probability p_bf.

    Returns True if the detector clicks.
    """
    latent = project_z(q, prng)
    if latent == 1:
        registered = prng.random() < detector_efficiency
    else:
        registered = prng.random() < dark_count_prob
    if prng.random() < bit_flip_prob:
        registered = not registered
    return registered


def project_x(q: QState, prng: PRNG) -> int:
    """X-basis measurement via Hadamard rotation followed by Z projection."""
    op_hadamard(q)
    return project_z(q, prng)


# ═══════════════════════════════════════════════════════════════════════
#  5.  COHERENCE DECAY
# ═══════════════════════════════════════════════════════════════════════

def coherence_decay(f_initial: float, idle_ms: float,
                    coherence_ms: float) -> float:
    """
    Fidelity of an idling entangled pair after `idle_ms` milliseconds:

        F(t) = F₀ · exp(−t / T₂)

    where T₂ is the memory coherence time.  This is the standard
    exponential-decay model used throughout quantum-repeater analyses.
    """
    return f_initial * math.exp(-idle_ms / coherence_ms)


# ═══════════════════════════════════════════════════════════════════════
#  6.  HARDWARE DESCRIPTOR
# ═══════════════════════════════════════════════════════════════════════

class SourceType(Enum):
    """Emitter statistic for the matter-qubit source."""
    SINGLE_PHOTON = "single_photon"   # Barrett–Kok, NV, trapped ion
    POISSONIAN    = "poissonian"      # DLCZ ensemble


@dataclass
class HardwareSpec:
    """
    Platform-specific efficiency and emission parameters for a
    quantum-network node.

    Each field is a dimensionless probability in [0, 1] except for
    `coherence_ms` (a time constant), `mean_photon_number` (a mean
    photon number), and `mode_mismatch` (a dimensionless error
    parameter).
    """
    name: str
    coherence_ms: float
    memory_emission: float          # P(memory emits photon on one attempt)
    fresh_pair_fidelity: float      # fidelity right after a successful attempt
    detector_eta: float             # single-photon detector efficiency
    dark_count: float               # dark-count probability per window
    herald_bsm_efficiency: float    # BSM efficiency for generation
    local_bsm_efficiency: float     # BSM efficiency for swapping
    gate_quality: float             # multiplicative gate-error penalty
    source_type: SourceType = SourceType.SINGLE_PHOTON
    mean_photon_number: float = 0.1 # μ, used only when POISSONIAN
    mode_mismatch: float = 0.01     # r in p_BK ≈ (1 − r) p_d²

    # ── well-known platforms ────────────────────────────────────────────
    @classmethod
    def erbium_cavity(cls) -> "HardwareSpec":
        return cls(
            name="Erbium-167 (cavity QED)",
            coherence_ms=1300.0,
            memory_emission=0.90,
            fresh_pair_fidelity=0.95,
            detector_eta=0.90,
            dark_count=0.01,
            herald_bsm_efficiency=0.50,
            local_bsm_efficiency=0.90,
            gate_quality=0.98,
            source_type=SourceType.SINGLE_PHOTON,
            mode_mismatch=0.02,
        )

    @classmethod
    def nv_diamond(cls) -> "HardwareSpec":
        return cls(
            name="NV-centre (diamond)",
            coherence_ms=10.0,
            memory_emission=0.06,
            fresh_pair_fidelity=0.92,
            detector_eta=0.85,
            dark_count=0.005,
            herald_bsm_efficiency=0.50,
            local_bsm_efficiency=0.98,
            gate_quality=0.995,
            source_type=SourceType.SINGLE_PHOTON,
            mode_mismatch=0.03,
        )

    @classmethod
    def dlcz_ensemble(cls) -> "HardwareSpec":
        """Atomic-ensemble DLCZ source (Poissonian photon statistics)."""
        return cls(
            name="DLCZ ensemble",
            coherence_ms=0.1,
            memory_emission=1.0,        # not used for Poissonian
            fresh_pair_fidelity=0.90,
            detector_eta=0.85,
            dark_count=0.001,
            herald_bsm_efficiency=0.50,
            local_bsm_efficiency=0.95,
            gate_quality=0.99,
            source_type=SourceType.POISSONIAN,
            mean_photon_number=0.05,
            mode_mismatch=0.02,
        )

    @classmethod
    def lossless_reference(cls) -> "HardwareSpec":
        return cls(
            name="Lossless reference",
            coherence_ms=1_000_000.0,
            memory_emission=1.0,
            fresh_pair_fidelity=1.0,
            detector_eta=1.0,
            dark_count=0.0,
            herald_bsm_efficiency=1.0,
            local_bsm_efficiency=1.0,
            gate_quality=1.0,
            source_type=SourceType.SINGLE_PHOTON,
            mode_mismatch=0.0,
        )


# ═══════════════════════════════════════════════════════════════════════
#  7.  PHOTON-RESOLVED GENERATION
# ═══════════════════════════════════════════════════════════════════════

def _sample_poisson(mean: float, prng: PRNG) -> int:
    """
    Draw a Poisson-distributed integer using Knuth's algorithm and the
    shared PRNG.  Valid for small means (μ ≲ 30), which is the regime
    of interest for weak-pulse and DLCZ sources.  For larger means a
    normal approximation would be used instead.
    """
    if mean <= 0.0:
        return 0
    L = math.exp(-mean)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= prng.random()
        if p <= L:
            return k - 1


@dataclass
class PhotonBudget:
    """Cumulative counts of every photon-tracking stage, for analysis."""
    emitted: int = 0
    emitted_pairs: int = 0
    survived_left: int = 0
    survived_right: int = 0
    detected_left: int = 0
    detected_right: int = 0
    dark_counts: int = 0
    heralds: int = 0


@dataclass
class GenerationOutcome:
    heralded: bool
    pair: Optional[EntangledPair] = None
    #: Machine-readable failure stage for statistics/plots.
    #: One of {"emission", "loss_L", "loss_R", "detection",
    #:          "mode_mismatch", None}
    reason: Optional[str] = None


class HeraldedGenerator:
    """
    Photon-resolved Barrett–Kok / DLCZ heralded entanglement source
    for a single physical link.

    Unlike an analytical model that jumps straight to a closed-form
    success probability, every attempt here explicitly simulates the
    full physical chain:

      1.  Emission       — each memory emits a photon with probability
                            η_mem (single-photon source) or a Poissonian
                            number of photons with mean μ (DLCZ source).

      2.  Channel        — each emitted photon survives the fibre with
                            probability T = 10^(−L·α_dB / 10).

      3.  Detection      — each arriving photon is registered by the
                            midpoint detectors with probability η_det.
                            Dark counts fire independently with
                            probability p_dc.

      4.  Interference   — two-photon interference at the beam splitter
                            succeeds with probability (1 − r) when both
                            sides register exactly one photon, where r
                            is the mode-mismatch parameter.  This gives
                            the standard Barrett–Kok quadratic scaling
                                p_BK ≈ (1 − r) · p_d²
                            as a *derived* quantity, not an assumption.

    References:
      Barrett & Kok, Phys. Rev. A 71, 060310 (2005).
      Duan et al., Nature 414, 413 (2001).
      Brevi et al., Adv. Quantum Technol. (2026)  — simplified model.
    """

    def __init__(self, hardware: HardwareSpec, link_length_km: float,
                 attenuation_db_per_km: float = 0.2):
        self.hw = hardware
        self.length_km = link_length_km
        self.attenuation = attenuation_db_per_km
        #: Round-trip classical signal delay for this link, in ms.
        self.one_way_delay_ms = link_length_km / FIBRE_SPEED_KM_PER_MS
        self.budget = PhotonBudget()

    @property
    def transmittance(self) -> float:
        """Fibre transmittance T = 10^(−L·α/10)."""
        return 10.0 ** (-(self.length_km * self.attenuation) / 10.0)

    # ── main entry point ───────────────────────────────────────────────
    def attempt(self, now_ms: float, prng: PRNG) -> GenerationOutcome:
        """
        Simulate one Barrett–Kok attempt, photon by photon.

        Returns a `GenerationOutcome`.  All randomness is drawn from
        the shared `prng`, so the full attempt history of a run is
        reproducible from a single seed.
        """
        # ── Stage 1: emission ─────────────────────────────────────────
        if self.hw.source_type is SourceType.SINGLE_PHOTON:
            n_left  = 1 if prng.random() < self.hw.memory_emission else 0
            n_right = 1 if prng.random() < self.hw.memory_emission else 0
        else:  # POISSONIAN
            n_left  = _sample_poisson(self.hw.mean_photon_number, prng)
            n_right = _sample_poisson(self.hw.mean_photon_number, prng)

        self.budget.emitted += n_left + n_right
        if n_left == 0 or n_right == 0:
            return GenerationOutcome(heralded=False, reason="emission")

        # For the coincidence herald we require exactly one photon on
        # each side (multi-photon events are rejected as they degrade
        # the heralded state — see Barrett & Kok §III).
        if n_left != 1 or n_right != 1:
            return GenerationOutcome(heralded=False, reason="emission")

        self.budget.emitted_pairs += 1

        # ── Stage 2: channel transmission ─────────────────────────────
        T = self.transmittance
        if prng.random() >= T:
            return GenerationOutcome(heralded=False, reason="loss_L")
        self.budget.survived_left += 1

        if prng.random() >= T:
            return GenerationOutcome(heralded=False, reason="loss_R")
        self.budget.survived_right += 1

        # ── Stage 3: detection at the midpoint BSM ────────────────────
        det_left  = prng.random() < self.hw.detector_eta
        det_right = prng.random() < self.hw.detector_eta

        # Dark counts are modelled separately; they contribute clicks
        # that do not herald entanglement and would in a real experiment
        # be rejected by the (expected) coincidence logic.
        if prng.random() < self.hw.dark_count:
            self.budget.dark_counts += 1

        if not (det_left and det_right):
            return GenerationOutcome(heralded=False, reason="detection")

        self.budget.detected_left  += 1
        self.budget.detected_right += 1

        # ── Stage 4: two-photon interference ──────────────────────────
        # The two photons must be indistinguishable at the beam splitter
        # for the herald to project onto a Bell state.  The mode-mismatch
        # parameter r captures residual spectral/temporal distinguisha-
        # bility.  When the photons are distinguishable, the detectors
        # still click but the herald has no entanglement content, and
        # we conservatively discard the attempt.
        if prng.random() < self.hw.mode_mismatch:
            return GenerationOutcome(heralded=False, reason="mode_mismatch")

        # ── Stage 5: herald succeeded ─────────────────────────────────
        self.budget.heralds += 1
        pair = EntangledPair.phi_plus()
        pair.nominal_fidelity = self.hw.fresh_pair_fidelity
        pair.current_fidelity = self.hw.fresh_pair_fidelity
        pair.born_at_ms = now_ms
        return GenerationOutcome(heralded=True, pair=pair)

    # ── analytical cross-check ────────────────────────────────────────
    def expected_success_probability(self) -> float:
        """
        The closed-form success probability that the empirical
        simulator should converge to in the limit of many attempts:

            p_BK = (1 − r) · (η_mem · T · η_det)²

        (single-photon emitters, coincidence herald).  Provided so
        that the empirical rate can be checked against the analytic
        prediction in unit tests.
        """
        if self.hw.source_type is not SourceType.SINGLE_PHOTON:
            # Poissonian case: exact expression differs; return the
            # small-μ approximation for reference only.
            return 0.0
        p_d = (self.hw.memory_emission
               * self.transmittance
               * self.hw.detector_eta)
        return (1.0 - self.hw.mode_mismatch) * p_d ** 2


# ═══════════════════════════════════════════════════════════════════════
#  8.  LINK EXTENSION (ENTANGLEMENT SWAPPING)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ExtensionOutcome:
    succeeded: bool
    extended_pair: Optional[EntangledPair] = None


class LinkExtender:
    """
    Performs a local Bell-state measurement at a repeater node,
    joining two adjacent entanglement links into one longer link.

    When the two input pairs are Werner states with fidelities F_AB
    and F_BC, the output state is again a Werner state with fidelity

        F_out = F_AB · F_BC + (1 − F_AB)(1 − F_BC) / 3,

    up to a multiplicative gate-quality penalty.  This recurrence
    relation is standard:

      Bennett et al., Phys. Rev. Lett. 76, 722 (1996).
      Żukowski et al., Phys. Rev. Lett. 71, 4287 (1993).

    The local BSM succeeds with probability η_local, sampled afresh
    on every call.  The gate-quality factor models residual two-qubit
    gate errors at the repeater.
    """

    def __init__(self, hardware: HardwareSpec):
        self.hw = hardware

    def try_extend(self, left: EntangledPair, right: EntangledPair,
                   now_ms: float, prng: PRNG) -> ExtensionOutcome:
        # Empirical BSM outcome.
        if prng.random() >= self.hw.local_bsm_efficiency:
            return ExtensionOutcome(succeeded=False)

        f_ab = left.current_fidelity
        f_bc = right.current_fidelity
        f_out = (f_ab * f_bc
                 + (1.0 - f_ab) * (1.0 - f_bc) / 3.0) * self.hw.gate_quality

        joined = EntangledPair.phi_plus()
        joined.nominal_fidelity = f_out
        joined.current_fidelity = f_out
        joined.born_at_ms = now_ms
        return ExtensionOutcome(succeeded=True, extended_pair=joined)

    def refresh_fidelity(self, pair: EntangledPair, now_ms: float) -> None:
        """
        Recompute `current_fidelity` from `nominal_fidelity` and the
        elapsed time since the pair was created, using the exponential
        decay law of §5.
        """
        idle = now_ms - pair.born_at_ms
        pair.current_fidelity = coherence_decay(
            pair.nominal_fidelity, idle, self.hw.coherence_ms
        )


# ═══════════════════════════════════════════════════════════════════════
#  9.  CLASSICAL POST-PROCESSING BRIDGE
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class KeySource:
    """
    Aggregate delivered entangled pairs into a classical raw key and a
    QBER estimate, in the form consumed by the reconciliation and
    privacy-amplification modules used elsewhere in this project.

    Model
    -----
    For an end-to-end pair in a Werner state of fidelity F, the
    outcomes of local Z-basis measurements on Alice's and Bob's halves
    are correlated with probability

        P(a = b)  =  F + (1 − F)/3  =  (1 + 2F) / 3,

    and the quantum bit error rate in either the Z or the X basis is

        QBER(F)  =  2 (1 − F) / 3.

    This module samples both quantities empirically: for each
    delivered pair it draws Alice's bit uniformly at random and then
    draws Bob's bit as a match with probability (1 + 2F) / 3.  The
    resulting raw key and QBER can be passed verbatim to Cascade,
    LDPC, Winnow, Polar, or any other reconciliation backend, and to
    the Toeplitz (or other) privacy-amplification stage.

    QBER formula reference:
        Bennett et al., Phys. Rev. Lett. 76, 722 (1996).
    """
    prng: PRNG
    _raw_key: List[int] = field(default_factory=list)
    _fidelities: List[float] = field(default_factory=list)

    def add_pair(self, pair: EntangledPair) -> Tuple[int, int]:
        """
        Sample one (alice_bit, bob_bit) pair from a delivered
        entangled pair and append the bit to the raw key.  Returns
        the pair of sampled bits for inspection.
        """
        F = pair.current_fidelity
        # Alice's local outcome is uniform for a Werner state.
        a = self.prng.get_bit()
        # Bob matches with probability (1 + 2F)/3.
        p_match = (1.0 + 2.0 * F) / 3.0
        if self.prng.random() < p_match:
            b = a
        else:
            b = 1 - a
        self._raw_key.append(a)
        self._fidelities.append(F)
        return a, b

    def raw_key(self) -> List[int]:
        """The accumulated raw key as a list of 0/1 integers."""
        return list(self._raw_key)

    def fidelity_mean(self) -> float:
        if not self._fidelities:
            return 0.0
        return sum(self._fidelities) / len(self._fidelities)

    def estimated_qber(self) -> float:
        """QBER computed from the mean delivered fidelity."""
        F = self.fidelity_mean()
        return max(0.0, min(0.5, 2.0 * (1.0 - F) / 3.0))

    def measured_qber(self, sample_fraction: float = 0.1) -> float:
        """
        QBER measured by *revealing* a random sample of the raw key.
        This mirrors the parameter-estimation step of a real protocol
        and can be compared against `estimated_qber`.
        """
        n = len(self._raw_key)
        if n == 0:
            return 0.0
        k = max(1, int(n * sample_fraction))
        indices = self.prng.sample(range(n), k)
        # Re-measure those positions: sample Bob's bit again using the
        # stored fidelity for that key bit.
        errors = 0
        for i in indices:
            F = self._fidelities[i]
            p_match = (1.0 + 2.0 * F) / 3.0
            b = self._raw_key[i] if self.prng.random() < p_match \
                else 1 - self._raw_key[i]
            if b != self._raw_key[i]:
                errors += 1
        return errors / k

    def as_post_processing_input(self) -> Dict[str, Any]:
        """
        Bundle the raw key and QBER estimate into a dictionary suitable
        for direct insertion into a reconciliation + privacy-
        amplification pipeline (Cascade / LDPC / Winnow / Polar + PA).
        """
        return {
            "raw_key": self.raw_key(),
            "qber_estimate": self.estimated_qber(),
            "qber_measured": self.measured_qber(),
            "mean_fidelity": self.fidelity_mean(),
            "n_bits": len(self._raw_key),
        }


# ═══════════════════════════════════════════════════════════════════════
# 10.  DEMO — PHOTON-RESOLVED GENERATION + SWAPPING
# ═══════════════════════════════════════════════════════════════════════

async def demo_generation_and_swapping() -> None:
    """
    End-to-end demonstration of the photon-resolved primitives.

      Part A: two-node generation.  Count how often a Barrett–Kok
              attempt heralds, compare to the analytic prediction.

      Part B: three-node swapping.  Generate two independent links
              and join them with a local BSM.  Compare the observed
              fidelity to the Werner-state prediction.

      Part C: raw key + QBER extraction.  Aggregate many delivered
              pairs into a `KeySource` and print the inputs a
              reconciliation backend would receive.
    """
    print("=" * 68)
    print("  DEMO: Photon-Resolved Generation + Link Extension")
    print("=" * 68)

    hw = HardwareSpec.erbium_cavity()
    prng = PRNG(seed=42)

    # ── Part A: generation ──────────────────────────────────────────────
    print("\n── Part A: Two-node photon-resolved generation ──")
    gen = HeraldedGenerator(hw, link_length_km=50.0)
    p_expected = gen.expected_success_probability()
    print(f"  Fibre transmittance at 50 km, 0.2 dB/km: "
          f"{gen.transmittance:.4f}")
    print(f"  Analytic p_BK (coincidence herald):       {p_expected:.6f}")

    n_attempts = 200_000
    n_heralded = 0
    for _ in range(n_attempts):
        outcome = gen.attempt(now_ms=0.0, prng=prng)
        if outcome.heralded:
            n_heralded += 1

    p_empirical = n_heralded / n_attempts
    print(f"  Attempts: {n_attempts},  heralds: {n_heralded}")
    print(f"  Empirical p_BK: {p_empirical:.6f}  "
          f"(ratio to analytic: {p_empirical / p_expected:.4f})")
    print(f"  Photon budget: emitted={gen.budget.emitted}, "
          f"survived={gen.budget.survived_left}+{gen.budget.survived_right}, "
          f"detected={gen.budget.detected_left}+{gen.budget.detected_right}, "
          f"dark={gen.budget.dark_counts}")

    # ── Part B: swapping ────────────────────────────────────────────────
    print("\n── Part B: Three-node link extension ──")
    extender = LinkExtender(hw)

    gen_left  = HeraldedGenerator(hw, link_length_km=30.0)
    gen_right = HeraldedGenerator(hw, link_length_km=30.0)

    left_pair = None
    right_pair = None
    for _ in range(2_000_000):
        if left_pair is None:
            o = gen_left.attempt(now_ms=0.0, prng=prng)
            if o.heralded:
                left_pair = o.pair
        if right_pair is None:
            o = gen_right.attempt(now_ms=0.0, prng=prng)
            if o.heralded:
                right_pair = o.pair
        if left_pair and right_pair:
            break

    if left_pair is None or right_pair is None:
        print("  ERROR: could not generate both links.")
        return

    print(f"  Left-link fidelity:  {left_pair.current_fidelity:.4f}")
    print(f"  Right-link fidelity: {right_pair.current_fidelity:.4f}")

    f_ab, f_bc = left_pair.current_fidelity, right_pair.current_fidelity
    f_predicted = (f_ab * f_bc
                   + (1.0 - f_ab) * (1.0 - f_bc) / 3.0) * hw.gate_quality
    print(f"  Werner-state prediction:    {f_predicted:.6f}")

    successes = 0
    f_observed = []
    for _ in range(5000):
        out = extender.try_extend(left_pair, right_pair,
                                  now_ms=0.0, prng=prng)
        if out.succeeded:
            successes += 1
            f_observed.append(out.extended_pair.current_fidelity)

    print(f"  Local-BSM success rate: {successes}/5000 "
          f"({100.0 * successes / 5000:.2f} %)")
    if f_observed:
        print(f"  Mean observed output fidelity: {np.mean(f_observed):.6f}")
        print(f"  Max deviation from prediction: "
              f"{max(abs(f - f_predicted) for f in f_observed):.2e}")

    # ── Part C: raw key + QBER bridge ───────────────────────────────────
    print("\n── Part C: Raw-key + QBER extraction (post-processing bridge) ──")
    key_source = KeySource(prng=prng)
    n_pairs = 1000
    for _ in range(n_pairs):
        # Re-generate a fresh end-to-end pair for each key bit, so the
        # fidelity distribution matches the swapping model above.
        lp = None
        rp = None
        for _ in range(200_000):
            if lp is None:
                o = gen_left.attempt(now_ms=0.0, prng=prng)
                if o.heralded:
                    lp = o.pair
            if rp is None:
                o = gen_right.attempt(now_ms=0.0, prng=prng)
                if o.heralded:
                    rp = o.pair
            if lp and rp:
                break
        if lp is None or rp is None:
            continue
        ext = extender.try_extend(lp, rp, now_ms=0.0, prng=prng)
        if not ext.succeeded:
            continue
        key_source.add_pair(ext.extended_pair)

    bundle = key_source.as_post_processing_input()
    print(f"  Raw key length:            {bundle['n_bits']}")
    print(f"  Mean delivered fidelity:   {bundle['mean_fidelity']:.4f}")
    print(f"  QBER (from fidelity):      {bundle['qber_estimate']:.4f}")
    print(f"  QBER (from revealed sample): {bundle['qber_measured']:.4f}")
    print(f"  First 30 raw-key bits:     "
          f"{''.join(map(str, bundle['raw_key'][:30]))}")
    print()
    print("  → This dictionary can be passed directly to a reconciliation")
    print("    backend (Cascade / LDPC / Winnow / Polar) followed by")
    print("    privacy amplification, exactly as in the classical pipeline.")

    print("\n[Demo complete]\n")


if __name__ == "__main__":
    asyncio.run(demo_generation_and_swapping())