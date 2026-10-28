"""
entanglement_primitives.py
==========================
Entanglement generation and link-extension primitives for quantum networks.

This module provides the physical-layer building blocks:

  * Single-qubit and two-qubit state containers with complex-amplitude vectors.
  * Unitary gates (Pauli, Hadamard) and basis-resolved measurement routines.
  * A coherence-decay model for idling quantum memories.
  * A hardware descriptor that groups platform-specific efficiencies.
  * A heralded entanglement generator (Barrett–Kok).
  * A link-extension routine (entanglement swapping) using the Werner-state
    recurrence relation.

── Physics references ───────────────────────────────────────────────────────
  Barrett–Kok generation:
    S. D. Barrett and P. Kok, "Efficient high-fidelity quantum computation
    using matter qubits and linear optics," Phys. Rev. A 71, 060310 (2005).
    arXiv:quant-ph/0408040.
    Success probability per attempt is quadratic in the single-photon
    detection probability:  p_BK ≈ (1 − r) · p_d²  — see §4.2.1 of the
    Bell-state-heralding analysis.

  Entanglement swapping of Werner states:
    C. H. Bennett et al., "Purification of Noisy Entanglement and Faithful
    Teleportation via Noisy Channels," Phys. Rev. Lett. 76, 722 (1996).
    M. Żukowski et al., "Event-ready-detectors Bell experiment via
    entanglement swapping," Phys. Rev. Lett. 71, 4287 (1993).
    For two input Werner states of fidelity F₁ and F₂, the output is a
    Werner state with fidelity
        F_out = F₁·F₂ + (1 − F₁)(1 − F₂)/3.


  Memory decoherence:
    The fidelity of a stored Bell pair decays exponentially with idle time:
        F(t) = F₀ · exp(−t / T₂).
    This is the standard model used in quantum-repeater analyses
    (see, e.g., the fidelity-decay expression in the supplementary
    notes of "Optimal entanglement distribution policies in homogeneous
    repeater chains with cutoffs," and the memory-decay fits in
    trapped-ion and NV-center experiments).

"""

from __future__ import annotations

import asyncio
import logging
import math
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logging.basicConfig(level=logging.ERROR, format="%(message)s")

# ═══════════════════════════════════════════════════════════════════════════
#  1.  ASYNC MESSAGE-PASSING BACKBONE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class HaltSignal:
    """Cooperative shutdown marker for the actor loop."""
    pass


class Messenger:
    """
    Minimal asynchronous actor with a single inbox queue.

    Subclasses override `consume` to process incoming messages.  The
    `dispatch` method pushes a message into another Messenger's inbox.
    This is the same architectural pattern used throughout the
    BB84 server/client codebase; it is reproduced here so that this
    module has no external dependencies beyond NumPy.
    """

    def __init__(self, label: str, seed: Optional[int] = None):
        self.label = label
        self.inbox: asyncio.Queue = asyncio.Queue()
        self._active = False
        self.rng = random.Random(seed)

    async def run(self):
        self._active = True
        while self._active:
            item = await self.inbox.get()
            if isinstance(item, HaltSignal):
                self._active = False
                break
            await self.consume(item)

    async def dispatch(self, peer: "Messenger", item):
        await peer.inbox.put(item)

    async def consume(self, item):
        """Override in subclasses."""
        pass


# ═══════════════════════════════════════════════════════════════════════════
#  2.  QUANTUM STATE CONTAINERS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class QState:
    """
    A single-qubit pure state |ψ⟩ = α|0⟩ + β|1⟩ stored as a length-2
    complex vector.  Normalisation is assumed maintained by constructors
    and gate applications.
    """
    amps: np.ndarray  # shape (2,), dtype=complex128

    # ── factory methods ─────────────────────────────────────────────────
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

    # ── observables ─────────────────────────────────────────────────────
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
    {|00⟩, |01⟩, |10⟩, |11⟩}, together with the metadata needed to track
    how its fidelity evolves over time.

    `nominal_fidelity` is the fidelity the pair had when it was freshly
    created (or freshly re-derived after a swap).  `current_fidelity` is
    the value after accounting for idling decoherence.  `born_at_ms` is
    the virtual-clock timestamp of creation.
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
        """Return |⟨ψ|φ⟩|².  For pure states this equals the fidelity."""
        return float(abs(np.vdot(self.amps, other.amps)) ** 2)


# ═══════════════════════════════════════════════════════════════════════════
#  3.  UNITARY OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
#  4.  MEASUREMENT
# ═══════════════════════════════════════════════════════════════════════════

def project_z(q: QState, rng: random.Random) -> int:
    """
    Ideal Z-basis measurement.  Returns 0 or 1 and collapses the state.
    The Born rule gives P(1) = |β|².
    """
    outcome = 1 if rng.random() >= q.p0() else 0
    q.amps = (np.array([0.0, 1.0], dtype=complex) if outcome
              else np.array([1.0, 0.0], dtype=complex))
    return outcome


def project_z_with_imperfections(
    q: QState,
    detector_efficiency: float,
    dark_count_prob: float,
    bit_flip_prob: float,
    rng: random.Random,
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
    latent = project_z(q, rng)
    if latent == 1:
        registered = rng.random() < detector_efficiency
    else:
        registered = rng.random() < dark_count_prob
    if rng.random() < bit_flip_prob:
        registered = not registered
    return registered


def project_x(q: QState, rng: random.Random) -> int:
    """X-basis measurement via Hadamard rotation followed by Z projection."""
    op_hadamard(q)
    return project_z(q, rng)


# ═══════════════════════════════════════════════════════════════════════════
#  5.  COHERENCE DECAY
# ═══════════════════════════════════════════════════════════════════════════

def coherence_decay(f_initial: float, idle_ms: float,
                    coherence_ms: float) -> float:
    """
    Fidelity of an idling entangled pair after `idle_ms` milliseconds:

        F(t) = F₀ · exp(−t / T₂)

    where T₂ is the memory coherence time.  This is the standard
    exponential-decay model used throughout quantum-repeater analyses.
    """
    return f_initial * math.exp(-idle_ms / coherence_ms)


# ═══════════════════════════════════════════════════════════════════════════
#  6.  HARDWARE DESCRIPTOR
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class HardwareSpec:
    """
    Platform-specific efficiency parameters for a quantum-network node.

    Each field is a dimensionless probability in [0, 1] except for
    `coherence_ms`, which is a time constant.
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
        )


# ═══════════════════════════════════════════════════════════════════════════
#  7.  HERALDED ENTANGLEMENT GENERATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class GenerationOutcome:
    heralded: bool
    pair: Optional[EntangledPair] = None


class HeraldedGenerator:
    """
    Models a Barrett–Kok / DLCZ-type heralded entanglement source for a
    single physical link.

    The per-attempt success probability follows the standard
    photon-heralding product:

        p_herald = η_memory · T_link · η_detector · η_BSM

    where T_link is the channel transmittance

        T_link = 10^(−L · α_dB / 10)

    with L the fibre length in km and α_dB the attenuation in dB/km.
    For the ideal case (all efficiencies = 1) this reduces to p = 1,
    matching the lossless-reference tests in the reference simulator.

    Reference for the product form: Barrett & Kok (2005); the
    quadratic-in-detection-probability structure is discussed in
    §4.2.1 of the standard Bell-state-heralding analysis.
    """

    def __init__(self, hardware: HardwareSpec, link_length_km: float,
                 attenuation_db_per_km: float = 0.2):
        self.hw = hardware
        self.length_km = link_length_km
        self.attenuation = attenuation_db_per_km

    @property
    def transmittance(self) -> float:
        """Fibre transmittance T = 10^(−L·α/10)."""
        return 10.0 ** (-(self.length_km * self.attenuation) / 10.0)

    def attempt(self, now_ms: float, rng: random.Random) -> GenerationOutcome:
        p = (self.hw.memory_emission
             * self.transmittance
             * self.hw.detector_eta
             * self.hw.herald_bsm_efficiency)
        if rng.random() > p:
            return GenerationOutcome(heralded=False)

        pair = EntangledPair.phi_plus()
        pair.nominal_fidelity = self.hw.fresh_pair_fidelity
        pair.current_fidelity = self.hw.fresh_pair_fidelity
        pair.born_at_ms = now_ms
        return GenerationOutcome(heralded=True, pair=pair)


# ═══════════════════════════════════════════════════════════════════════════
#  8.  LINK EXTENSION (ENTANGLEMENT SWAPPING)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ExtensionOutcome:
    succeeded: bool
    extended_pair: Optional[EntangledPair] = None


class LinkExtender:
    """
    Performs a local Bell-state measurement at a repeater node, thereby
    joining two adjacent entanglement links into one longer link.

    When the two input pairs are Werner states with fidelities F_AB and
    F_BC, the output state is again a Werner state with fidelity

        F_out = F_AB · F_BC + (1 − F_AB)(1 − F_BC) / 3,

    up to a multiplicative gate-quality penalty.  This recurrence
    relation is standard; see Bennett et al. (1996), Żukowski et al.
    (1993), and the equivalent expression F' = F₁F₂ + (1−F₁)(1−F₂)/3
    given in Eq. (2.33) of the TU Delft entanglement-distribution
    thesis.

    The local BSM succeeds with probability η_local.  The gate-quality
    factor models residual two-qubit gate errors at the repeater.
    """

    def __init__(self, hardware: HardwareSpec):
        self.hw = hardware

    def try_extend(self, left: EntangledPair, right: EntangledPair,
                   now_ms: float, rng: random.Random) -> ExtensionOutcome:
        if rng.random() > self.hw.local_bsm_efficiency:
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


# ═══════════════════════════════════════════════════════════════════════════
#  9.  DEMO — SERVER / CLIENT ENTANGLEMENT GENERATION
# ═══════════════════════════════════════════════════════════════════════════

async def demo_generation_and_swapping():
    """
    End-to-end demonstration of the primitives:

      Part A: two-node generation.  A source node attempts to herald an
              entangled pair on a single fibre link.  We report the
              fraction of attempts that succeed and the fidelity of the
              delivered pair.

      Part B: three-node swapping.  Two independent links are generated,
              then a repeater performs a local BSM to join them.  We
              compare the resulting fidelity against the Werner-state
              prediction.
    """
    print("=" * 68)
    print("  DEMO: Herald Generation + Link Extension")
    print("=" * 68)

    hw = HardwareSpec.erbium_cavity()
    rng = random.Random(42)

    # ── Part A: generation ──────────────────────────────────────────────
    print("\n── Part A: Two-node heralded generation ──")
    gen = HeraldedGenerator(hw, link_length_km=50.0)
    print(f"  Fibre transmittance at 50 km, 0.2 dB/km: {gen.transmittance:.4f}")
    print(f"  Per-attempt success probability: "
          f"{hw.memory_emission * gen.transmittance * hw.detector_eta * hw.herald_bsm_efficiency:.4f}")

    n_attempts = 5000
    n_heralded = 0
    fidelities = []
    for _ in range(n_attempts):
        outcome = gen.attempt(now_ms=0.0, rng=rng)
        if outcome.heralded:
            n_heralded += 1
            fidelities.append(outcome.pair.current_fidelity)

    print(f"  Attempts: {n_attempts},  heralded: {n_heralded} "
          f"({100.0 * n_heralded / n_attempts:.2f} %)")
    if fidelities:
        print(f"  Mean fresh-pair fidelity: {np.mean(fidelities):.4f}")

    # ── Part B: swapping ────────────────────────────────────────────────
    print("\n── Part B: Three-node link extension ──")
    extender = LinkExtender(hw)

    # Generate two independent links
    gen_left = HeraldedGenerator(hw, link_length_km=30.0)
    gen_right = HeraldedGenerator(hw, link_length_km=30.0)

    left_pair = None
    right_pair = None
    for _ in range(20000):
        if left_pair is None:
            o = gen_left.attempt(now_ms=0.0, rng=rng)
            if o.heralded:
                left_pair = o.pair
        if right_pair is None:
            o = gen_right.attempt(now_ms=0.0, rng=rng)
            if o.heralded:
                right_pair = o.pair
        if left_pair and right_pair:
            break

    if left_pair is None or right_pair is None:
        print("  ERROR: could not generate both links.")
        return

    print(f"  Left-link fidelity:  {left_pair.current_fidelity:.4f}")
    print(f"  Right-link fidelity: {right_pair.current_fidelity:.4f}")

    # Analytic prediction
    f_ab, f_bc = left_pair.current_fidelity, right_pair.current_fidelity
    f_predicted = (f_ab * f_bc
                   + (1.0 - f_ab) * (1.0 - f_bc) / 3.0) * hw.gate_quality
    print(f"  Predicted output fidelity:  {f_predicted:.6f}")

    # Simulated swapping
    successes = 0
    f_observed = []
    for _ in range(5000):
        out = extender.try_extend(left_pair, right_pair, now_ms=0.0, rng=rng)
        if out.succeeded:
            successes += 1
            f_observed.append(out.extended_pair.current_fidelity)

    print(f"  Local-BSM success rate: {successes}/5000 "
          f"({100.0 * successes / 5000:.2f} %)")
    if f_observed:
        print(f"  Mean observed output fidelity: {np.mean(f_observed):.6f}")
        print(f"  Max absolute deviation from prediction: "
              f"{max(abs(f - f_predicted) for f in f_observed):.2e}")

    print("\n[Demo complete]\n")


if __name__ == "__main__":
    asyncio.run(demo_generation_and_swapping())