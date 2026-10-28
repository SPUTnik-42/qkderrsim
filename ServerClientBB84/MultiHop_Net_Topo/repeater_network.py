"""
repeater_network.py
===================
Multi-hop quantum repeater chain built on the primitives defined in
`entanglement_primitives.py`.

The chain is described by an ordered list of node labels and an ordered
list of link lengths.  Each node maintains a register of active
entanglement links.  Interior nodes attempt local Bell-state measurements
whenever they hold a link extending to the left of their position in the
chain and another extending to the right.

── Routing / scheduling references ──────────────────────────────────────────
  The "position-based left/right" rule for deciding when a repeater
  should perform a swap is the standard scheduling rule for linear
  repeater chains.  It is equivalent to the rule used in the reference
  discrete-event engine and is described in the quantum-repeater
  literature (see, e.g., the swap-ASAP analysis in "On noise in swap
  ASAP repeater chains" and the waiting-time analyses in "Efficient
  computation of the waiting time and fidelity in quantum repeater
  chains").

  Decoherence sweeps are performed at a fixed virtual-clock interval.
  The fidelity of each stored pair is refreshed from its nominal value
  using the exponential model of `coherence_decay`.

"""



from __future__ import annotations
import sys
from pathlib import Path
import time
import asyncio
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ServerClientBB84.entanglement_primitives import (
    Messenger,
    HaltSignal,
    EntangledPair,
    HardwareSpec,
    HeraldedGenerator,
    LinkExtender,
)




# ═══════════════════════════════════════════════════════════════════════════
#  1.  MESSAGES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LinkCreated:
    origin: int
    destination: int
    pair: EntangledPair


@dataclass
class ExtensionAnnouncement:
    repeater: int
    left_endpoint: int
    right_endpoint: int
    succeeded: bool
    joined_pair: Optional[EntangledPair]
    at_ms: float


@dataclass
class CoherenceSweep:
    now_ms: float


# ═══════════════════════════════════════════════════════════════════════════
#  2.  NODE
# ═══════════════════════════════════════════════════════════════════════════

class NetworkNode(Messenger):
    def __init__(self,
                 label: str,
                 index: int,
                 chain_position: int,
                 hardware: HardwareSpec,
                 settle_ms: float,
                 is_terminus_left: bool = False,
                 is_terminus_right: bool = False,
                 fidelity_floor: float = 0.5,
                 seed: Optional[int] = None):
        super().__init__(label, seed=seed)
        self.index = index
        self.position = chain_position
        self.hw = hardware
        self.settle_ms = settle_ms
        self.is_left_end = is_terminus_left
        self.is_right_end = is_terminus_right
        self.fidelity_floor = fidelity_floor

        self.active_links: Dict[int, EntangledPair] = {}

        self.adjacent: List[int] = []
        self.messaging_peers: Dict[int, "NetworkNode"] = {}
        self.chain_positions: Dict[int, int] = {}
        self.link_generators: Dict[int, HeraldedGenerator] = {}
        self.extender = LinkExtender(hardware)

        self._swap_in_flight = False
        self.clock_ms: float = 0.0

        self.heralds_attempted = 0
        self.heralds_succeeded = 0
        self.swaps_attempted = 0
        self.swaps_succeeded = 0
        self.evictions = 0
        self.delivered: List[float] = []

    # ── dispatch ────────────────────────────────────────────────────────
    async def consume(self, item):
        try:
            if isinstance(item, LinkCreated):
                await self._on_link_created(item)
            elif isinstance(item, ExtensionAnnouncement):
                await self._on_extension_announcement(item)
            elif isinstance(item, CoherenceSweep):
                await self._on_coherence_sweep(item)
        except Exception as exc:                        # noqa: BLE001
            import traceback
            print(f"[{self.label}] handler error on "
                  f"{type(item).__name__}: {exc}")
            traceback.print_exc()

    # ── generation ──────────────────────────────────────────────────────
    async def try_link_with(self, peer_index: int):
        if peer_index in self.active_links:
            return
        peer = self.messaging_peers.get(peer_index)
        if peer is None:
            return
        if self.index in peer.active_links:
            return
        gen = self.link_generators.get(peer_index)
        if gen is None:
            return

        self.heralds_attempted += 1
        outcome = gen.attempt(self.clock_ms, self.rng)
        if not outcome.heralded:
            return
        self.heralds_succeeded += 1

        self.active_links[peer_index] = outcome.pair
        await self.dispatch(peer, LinkCreated(
            origin=self.index, destination=peer_index, pair=outcome.pair))
        await self._maybe_extend()
        await peer._maybe_extend()

    async def _on_link_created(self, msg: LinkCreated):
        # Newest wins — no silent stale-pair retention.
        self.active_links[msg.origin] = msg.pair
        await self._maybe_extend()

    # ── link extension ──────────────────────────────────────────────────
    async def _maybe_extend(self):
        if self.is_left_end or self.is_right_end:
            return
        if self._swap_in_flight:
            return

        # Only pairs that have settled long enough may be consumed.
        eligible = {
            pid: p for pid, p in self.active_links.items()
            if (self.clock_ms - p.born_at_ms) >= self.settle_ms
        }
        if len(eligible) < 2:
            return

        left_partner: Optional[int] = None
        left_pos = -1
        right_partner: Optional[int] = None
        right_pos = float("inf")
        for pid in eligible:
            p = self.chain_positions.get(pid)
            if p is None:
                continue
            if p < self.position and p > left_pos:
                left_pos = p
                left_partner = pid
            elif p > self.position and p < right_pos:
                right_pos = p
                right_partner = pid

        if left_partner is None or right_partner is None:
            return

        self._swap_in_flight = True
        try:
            left_pair = self.active_links.pop(left_partner)
            right_pair = self.active_links.pop(right_partner)

            self.extender.refresh_fidelity(left_pair, self.clock_ms)
            self.extender.refresh_fidelity(right_pair, self.clock_ms)

            self.swaps_attempted += 1
            outcome = self.extender.try_extend(
                left_pair, right_pair, self.clock_ms, self.rng
            )
            if outcome.succeeded:
                self.swaps_succeeded += 1

            for pid in (left_partner, right_partner):
                target = self.messaging_peers.get(pid)
                if target is None:
                    continue
                await self.dispatch(target, ExtensionAnnouncement(
                    repeater=self.index,
                    left_endpoint=left_partner,
                    right_endpoint=right_partner,
                    succeeded=outcome.succeeded,
                    joined_pair=(outcome.extended_pair
                                 if outcome.succeeded else None),
                    at_ms=self.clock_ms,
                ))
        finally:
            self._swap_in_flight = False

    async def _on_extension_announcement(self, msg: ExtensionAnnouncement):
        self.active_links.pop(msg.repeater, None)
        if not msg.succeeded:
            return
        if self.index == msg.left_endpoint:
            other = msg.right_endpoint
        elif self.index == msg.right_endpoint:
            other = msg.left_endpoint
        else:
            return
        self.active_links[other] = msg.joined_pair
        await self._maybe_extend()

    # ── coherence ───────────────────────────────────────────────────────
    async def _on_coherence_sweep(self, msg: CoherenceSweep):
        self.clock_ms = msg.now_ms
        dead: List[int] = []
        for pid, pair in self.active_links.items():
            self.extender.refresh_fidelity(pair, msg.now_ms)
            if pair.current_fidelity < self.fidelity_floor:
                dead.append(pid)
        for pid in dead:
            del self.active_links[pid]
            self.evictions += 1


# ═══════════════════════════════════════════════════════════════════════════
#  3.  CHAIN
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ChainConfig:
    target_pairs: int = 10
    max_virtual_ms: float = 100_000.0
    retry_interval_ms: float = 1.0
    coherence_sweep_ms: float = 10.0
    fidelity_floor: float = 0.5
    settle_ms: float = 2.0              # classical herald propagation delay


class RepeaterChain:
    def __init__(self,
                 node_labels: List[str],
                 link_lengths_km: List[float],
                 hardware: HardwareSpec,
                 config: Optional[ChainConfig] = None,
                 attenuation_db_per_km: float = 0.2,
                 seed: Optional[int] = None):
        assert len(link_lengths_km) == len(node_labels) - 1
        self.n = len(node_labels)
        self.hw = hardware
        self.cfg = config or ChainConfig()
        self._root_rng = random.Random(seed)

        self._clock_ms = 0.0
        self._delivered_count = 0
        self._running = False
        self._tasks: List[asyncio.Task] = []

        positions = {i: i for i in range(self.n)}

        # ── nodes ────────────────────────────────────────────────────────
        self.nodes: List[NetworkNode] = []
        for i, label in enumerate(node_labels):
            node = NetworkNode(
                label=label, index=i, chain_position=i, hardware=hardware,
                settle_ms=self.cfg.settle_ms,
                is_terminus_left=(i == 0),
                is_terminus_right=(i == self.n - 1),
                fidelity_floor=self.cfg.fidelity_floor,
                seed=self._root_rng.randint(0, 2**31 - 1),
            )
            node.chain_positions = positions
            self.nodes.append(node)

        # ── all-to-all classical messaging ───────────────────────────────
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    self.nodes[i].messaging_peers[j] = self.nodes[j]

        # ── adjacent-only quantum generators ─────────────────────────────
        for i in range(self.n - 1):
            gen = HeraldedGenerator(hardware, link_lengths_km[i],
                                    attenuation_db_per_km)
            self.nodes[i].adjacent.append(i + 1)
            self.nodes[i + 1].adjacent.append(i)
            self.nodes[i].link_generators[i + 1] = gen
            self.nodes[i + 1].link_generators[i] = gen

    # ── main ────────────────────────────────────────────────────────────
    async def run(self):
        self._running = True
        for node in self.nodes:
            self._tasks.append(asyncio.create_task(node.run()))
        self._tasks.append(asyncio.create_task(self._clock_loop()))

        wall_deadline = time.monotonic() + 60.0
        try:
            while (self._delivered_count < self.cfg.target_pairs
                   and self._clock_ms < self.cfg.max_virtual_ms
                   and time.monotonic() < wall_deadline):
                await asyncio.sleep(0.001)
        finally:
            self._running = False
            for node in self.nodes:
                await node.dispatch(node, HaltSignal())
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._report()

    # ── clock ───────────────────────────────────────────────────────────
    async def _clock_loop(self):
        try:
            next_sweep = self.cfg.coherence_sweep_ms
            dt = max(self.cfg.retry_interval_ms, 1e-3)
            while self._running:
                # Push current virtual time to every node
                for node in self.nodes:
                    node.clock_ms = self._clock_ms

                # One generation attempt per adjacent link
                for i in range(self.n - 1):
                    await self.nodes[i].try_link_with(i + 1)

                # Re-evaluate swap eligibility for interior nodes
                # At most one swap per tick, processed left-to-right so the swap
                # wave propagates R1 → R2 → R3 → … through the chain.
                for i in range(1, self.n - 1):
                    node = self.nodes[i]
                    before = node.swaps_attempted
                    await node._maybe_extend()
                    if node.swaps_attempted > before:
                        break

                    
                # Delivery?
                await self._check_delivery()

                # Periodic coherence sweep
                if self._clock_ms >= next_sweep:
                    for node in self.nodes:
                        await node.dispatch(
                            node, CoherenceSweep(now_ms=self._clock_ms)
                        )
                    next_sweep += self.cfg.coherence_sweep_ms

                self._clock_ms += dt
                await asyncio.sleep(0)
        except Exception as exc:                          # noqa: BLE001
            import traceback
            print(f"[clock_loop] FATAL: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            self._running = False
            raise

    # ── delivery ────────────────────────────────────────────────────────
    async def _check_delivery(self):
        left = self.nodes[0]
        right = self.nodes[-1]
        if right.index not in left.active_links:
            return
        pair = left.active_links.pop(right.index)
        right.active_links.pop(left.index, None)
        left.delivered.append(pair.current_fidelity)
        self._delivered_count += 1

    # ── report ──────────────────────────────────────────────────────────
    def _report(self):
        gen_att = sum(n.heralds_attempted for n in self.nodes)
        gen_ok = sum(n.heralds_succeeded for n in self.nodes)
        sw_att = sum(n.swaps_attempted for n in self.nodes)
        sw_ok = sum(n.swaps_succeeded for n in self.nodes)
        evict = sum(n.evictions for n in self.nodes)
        fids = self.nodes[0].delivered

        print("\n" + "=" * 60)
        print("  REPEATER-CHAIN RESULTS")
        print("=" * 60)
        print(f"  Chain length:          {self.n} nodes")
        print(f"  Virtual clock elapsed: {self._clock_ms:.1f} ms")
        print(f"  Herald attempts:       {gen_att}")
        print(f"  Herald successes:      {gen_ok} "
              f"({100.0 * gen_ok / gen_att if gen_att else 0:.1f} %)")
        print(f"  Swap attempts:         {sw_att}")
        print(f"  Swap successes:        {sw_ok} "
              f"({100.0 * sw_ok / sw_att if sw_att else 0:.1f} %)")
        print(f"  Coherence evictions:   {evict}")
        print(f"  End-to-end deliveries: {self._delivered_count}")
        if fids:
            mean = sum(fids) / len(fids)
            var = (sum((f - mean) ** 2 for f in fids) / (len(fids) - 1)
                   if len(fids) > 1 else 0.0)
            print(f"  Mean delivered fidelity: {mean:.4f} ± {var ** 0.5:.4f}")
        print("=" * 60 + "\n")


# ═══════════════════════════════════════════════════════════════════════════
#  4.  DEMO
# ═══════════════════════════════════════════════════════════════════════════

async def demo_repeater_chain():
    print("\n" + "#" * 60)
    print("#  REPEATER NETWORK DEMONSTRATION")
    print("#" * 60)

    hw = HardwareSpec.erbium_cavity()

    scenarios = [
        ("2-node direct link",          ["A", "B"],                           [8.0],                   20),
        ("3-node chain (1 repeater)",   ["A", "R1", "B"],                     [8.0, 8.0],              20),
        ("4-node chain (2 repeaters)",  ["A", "R1", "R2", "B"],               [8.0, 8.0, 8.0],         15),
        ("5-node chain (3 repeaters)",  ["A", "R1", "R2", "R3", "B"],         [8.0]*4,                 10),
        ("6-node chain (4 repeaters)",  ["A", "R1", "R2", "R3", "R4", "B"],   [8.0]*5,                  8),
        ("7-node chain (5 repeaters)",  ["A", "R1", "R2", "R3", "R4", "R5", "B"], [8.0]*6,             6),
    ]

    means = []
    for idx, (label, labels, lengths, target) in enumerate(scenarios):
        print(f"\n>>> {label} <<<")
        cfg = ChainConfig(
            target_pairs=target,
            max_virtual_ms=200_000.0,
            retry_interval_ms=1.0,
            coherence_sweep_ms=10.0,
            settle_ms=2.0,
        )
        chain = RepeaterChain(
            node_labels=labels,
            link_lengths_km=lengths,
            hardware=hw,
            config=cfg,
            seed=100 * (idx + 1),
        )
        await chain.run()
        fids = chain.nodes[0].delivered
        means.append(sum(fids) / len(fids) if fids else 0.0)

    print("Mean delivered fidelity across chain lengths:")
    for (label, _, _, _), f in zip(scenarios, means):
        print(f"  {label:<40s} {f:.4f}")


if __name__ == "__main__":
    asyncio.run(demo_repeater_chain())