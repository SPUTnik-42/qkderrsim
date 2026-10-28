"""
repeater_topo.py
============================
Topology + routing layer for the multi-hop repeater simulator.

Two routing strategies are provided:

  * ShortestPathRouting — Dijkstra on the total edge-distance sum.
  * MaxRateRouting      — widest-path (minimax) on the per-link herald
                          probability, i.e. minimise the length of the
                          *longest* edge on the path.

The chosen path and its edge lengths are handed to `RepeaterChain`,
which simulates the physical layer on that exact chain.

Two demos are included:

  demo_convergent()  — the 6-node mesh from the BB84 multihop script.
                       Both strategies pick the same path.

  demo_divergent()   — a 5-node topology designed so the two strategies
                       pick *different* paths.  This isolates the
                       "total km" vs "worst single link" trade-off.
"""

from __future__ import annotations

import asyncio
import heapq
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ServerClientBB84.entanglement_primitives import HardwareSpec
from ServerClientBB84.MultiHop_Net_Topo.repeater_network import (
    RepeaterChain, ChainConfig,
)


# ═══════════════════════════════════════════════════════════════════════════
#  1.  PER-EDGE HERALD MODEL
# ═══════════════════════════════════════════════════════════════════════════

def herald_probability(hw: HardwareSpec,
                       distance_km: float,
                       attenuation_db_per_km: float) -> float:
    """
    Per-attempt herald probability on a link of length L:

        p_herald = η_mem · 10^(−L·α_dB/10) · η_det · η_BSM

    """
    T = 10.0 ** (-(distance_km * attenuation_db_per_km) / 10.0)
    return (hw.memory_emission * T * hw.detector_eta
            * hw.herald_bsm_efficiency)


# ═══════════════════════════════════════════════════════════════════════════
#  2.  ROUTING STRATEGIES  (only two, as requested)
# ═══════════════════════════════════════════════════════════════════════════

class RoutingStrategy(ABC):
    @abstractmethod
    def find_path(self,
                  graph: nx.Graph,
                  source: str,
                  target: str,
                  hw: HardwareSpec,
                  attenuation_db_per_km: float,
                  ) -> Tuple[List[str], float, float]:
        """
        Returns (path, total_distance_km, bottleneck_herald_prob).
        """
        ...


class ShortestPathRouting(RoutingStrategy):
    """
    Minimise the total physical distance along the path.

    Standard Dijkstra on the `distance` edge attribute.  The reported
    bottleneck herald probability is the *worst* link on the chosen
    path — included so the two strategies can be compared directly.
    """

    def find_path(self, graph, source, target, hw, attenuation_db_per_km):
        try:
            path = nx.shortest_path(graph, source=source, target=target,
                                    weight="distance")
        except nx.NetworkXNoPath:
            return [], 0.0, 0.0

        lengths = [graph[path[i]][path[i + 1]]["distance"]
                   for i in range(len(path) - 1)]
        p_bn = min(herald_probability(hw, L, attenuation_db_per_km)
                   for L in lengths)
        return path, sum(lengths), p_bn


class MaxRateRouting(RoutingStrategy):
    """
    Widest-path routing: minimise the *longest* single link on the path.

    Because herald probability is monotonically decreasing in distance,
    maximising the bottleneck p_herald is equivalent to minimising the
    maximum single-link distance.  Implemented as a minimax Dijkstra:
    the path metric is `max(edge lengths)`, not their sum.
    """

    def find_path(self, graph, source, target, hw, attenuation_db_per_km):
        pq: list = [(0.0, source, [source])]
        visited: set = set()
        best_path: List[str] = []
        best_max_d = float("inf")

        while pq:
            max_d, node, path = heapq.heappop(pq)

            if node == target:
                if max_d < best_max_d:
                    best_max_d = max_d
                    best_path = path
                continue

            if node in visited:
                continue
            visited.add(node)

            for nbr in graph.neighbors(node):
                d = graph[node][nbr]["distance"]
                heapq.heappush(pq, (max(max_d, d), nbr, path + [nbr]))

        if not best_path:
            return [], 0.0, 0.0

        lengths = [graph[best_path[i]][best_path[i + 1]]["distance"]
                   for i in range(len(best_path) - 1)]
        p_bn = herald_probability(hw, best_max_d, attenuation_db_per_km)
        return best_path, sum(lengths), p_bn


# ═══════════════════════════════════════════════════════════════════════════
#  3.  TOPOLOGY WRAPPER
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RoutedPath:
    nodes: List[str]
    link_lengths_km: List[float]
    total_distance_km: float
    bottleneck_herald_prob: float

    @property
    def n_hops(self) -> int:
        return len(self.nodes) - 1

    @property
    def bottleneck_km(self) -> float:
        return max(self.link_lengths_km) if self.link_lengths_km else 0.0


class RepeaterTopology:
    """
    Graph of trusted repeater stations connected by optical links of
    known length.  Routing selects an ordered path; `simulate()`
    instantiates a `RepeaterChain` on that path.
    """

    def __init__(self,
                 node_labels: List[str],
                 edges: List[Tuple[str, str, float]],
                 hardware: HardwareSpec,
                 attenuation_db_per_km: float = 0.2):
        self.graph = nx.Graph()
        self.graph.add_nodes_from(node_labels)
        for u, v, dist in edges:
            self.graph.add_edge(u, v, distance=dist)
        self.hardware = hardware
        self.attenuation = attenuation_db_per_km

    def route(self, source, target, strategy) -> RoutedPath:
        nodes, total_km, p_bn = strategy.find_path(
            self.graph, source, target,
            self.hardware, self.attenuation,
        )
        if len(nodes) < 2:
            raise ValueError(f"No route {source!r} → {target!r}")
        lengths = [self.graph[nodes[i]][nodes[i + 1]]["distance"]
                   for i in range(len(nodes) - 1)]
        return RoutedPath(
            nodes=nodes,
            link_lengths_km=lengths,
            total_distance_km=total_km,
            bottleneck_herald_prob=p_bn,
        )

    async def simulate(self, source, target, strategy,
                       config: ChainConfig, seed=None
                       ) -> Tuple[RoutedPath, RepeaterChain]:
        routed = self.route(source, target, strategy)
        chain = RepeaterChain(
            node_labels=routed.nodes,
            link_lengths_km=routed.link_lengths_km,
            hardware=self.hardware,
            config=config,
            attenuation_db_per_km=self.attenuation,
            seed=seed,
        )
        await chain.run()
        return routed, chain


# ═══════════════════════════════════════════════════════════════════════════
#  4.  DEMO A — CONVERGENT  (both routers pick the same path)
# ═══════════════════════════════════════════════════════════════════════════

CONVERGENT_NODES = ["Alice", "R1", "R2", "R3", "R4", "Bob"]
CONVERGENT_EDGES = [
    ("Alice", "R1",  50), ("Alice", "R2", 100),
    ("R1",    "R2",  30), ("R1",    "R3",  80),
    ("R2",    "R3",  40), ("R2",    "R4",  70),
    ("R3",    "Bob",100), ("R4",    "Bob", 40),
    ("R3",    "R4",  20),
]


async def demo_convergent():
    print("\n" + "#" * 74)
    print("#  DEMO A — CONVERGENT TOPOLOGY")
    print("#  Same 6-node mesh as the BB84 multihop analysis.")
    print("#  Both routers are expected to pick the same path.")
    print("#" * 74)

    hw = HardwareSpec.erbium_cavity()
    topo = RepeaterTopology(CONVERGENT_NODES, CONVERGENT_EDGES, hw)

    strategies = [
        ("ShortestPathRouting", ShortestPathRouting()),
        ("MaxRateRouting",      MaxRateRouting()),
    ]
    for name, strat in strategies:
        r = topo.route("Alice", "Bob", strat)
        print(f"\n  [{name}]")
        print(f"    Path:      {' → '.join(r.nodes)}")
        print(f"    Total km:  {r.total_distance_km:.0f}")
        print(f"    Max link:  {r.bottleneck_km:.0f} km  "
              f"(p_bn = {r.bottleneck_herald_prob:.4f})")


# ═══════════════════════════════════════════════════════════════════════════
#  5.  DEMO B — DIVERGENT  (two routers disagree)
# ═══════════════════════════════════════════════════════════════════════════
#
#  Topology designed to force a routing split:
#
#        Alice ──5 km── R1 ──100 km── Bob     ← short total, awful bottleneck
#          │
#          │ 40 km
#          ▼
#         R2 ──40 km── R3 ──40 km── Bob        ← longer total, uniform 40 km
#
#  * ShortestPathRouting:  Alice → R1 → Bob       total 105 km, bottleneck 100 km
#  * MaxRateRouting:       Alice → R2 → R3 → Bob  total 120 km, bottleneck  40 km
#
DIVERGENT_NODES = ["Alice", "R1", "R2", "R3", "Bob"]
DIVERGENT_EDGES = [
    ("Alice", "R1",   5.0),
    ("R1",    "Bob", 100.0),
    ("Alice", "R2",  40.0),
    ("R2",    "R3",  40.0),
    ("R3",    "Bob", 40.0),
]


async def demo_divergent():
    print("\n" + "#" * 74)
    print("#  DEMO B — DIVERGENT TOPOLOGY")
    print("#  Shortest-total-km path contains a 100 km link.")
    print("#  A slightly longer path avoids it (all 40 km links).")
    print("#" * 74)

    hw = HardwareSpec.erbium_cavity()
    topo = RepeaterTopology(DIVERGENT_NODES, DIVERGENT_EDGES, hw)

    # ── Step 1: analytic routing comparison ────────────────────────────
    print("\n─── Step 1: Analytic routing comparison ───\n")
    strategies = [
        ("ShortestPathRouting", ShortestPathRouting()),
        ("MaxRateRouting",      MaxRateRouting()),
    ]
    routed_by_name: Dict[str, RoutedPath] = {}

    print(f"  {'Strategy':<22s} {'Path':<30s} "
          f"{'Total km':>9s} {'Max link km':>12s} {'p_bn':>10s}")
    print("  " + "-" * 86)
    for name, strat in strategies:
        r = topo.route("Alice", "Bob", strat)
        routed_by_name[name] = r
        print(f"  {name:<22s} {' → '.join(r.nodes):<30s} "
              f"{r.total_distance_km:>9.0f} {r.bottleneck_km:>12.0f} "
              f"{r.bottleneck_herald_prob:>10.4f}")

    p_sp = routed_by_name["ShortestPathRouting"].bottleneck_herald_prob
    p_mr = routed_by_name["MaxRateRouting"].bottleneck_herald_prob
    if p_sp > 0:
        print(f"\n  Bottleneck herald-rate advantage of MaxRate path: "
              f"{p_mr / p_sp:.1f}×  (per attempt)")

    # ── Step 2: run the repeater chain on each selected path ──────────
    print("\n─── Step 2: Repeater-chain simulation on each path ───\n")
    cfg = ChainConfig(
        target_pairs=3,
        max_virtual_ms=500_000.0,
        retry_interval_ms=1.0,
        coherence_sweep_ms=10.0,
        settle_ms=2.0,
    )

    results = {}
    for name, strat in strategies:
        print(f">>> Simulating path chosen by {name} <<<")
        routed, chain = await topo.simulate(
            source="Alice", target="Bob",
            strategy=strat, config=cfg, seed=7,
        )
        fids = chain.nodes[0].delivered
        fbar = sum(fids) / len(fids) if fids else 0.0
        results[name] = dict(
            nodes=routed.nodes,
            total_km=routed.total_distance_km,
            bn_km=routed.bottleneck_km,
            p_bn=routed.bottleneck_herald_prob,
            delivered=len(fids),
            fbar=fbar,
            virtual_ms=chain._clock_ms,
        )

    # ── Step 3: side-by-side summary ──────────────────────────────────
    print("\n" + "=" * 90)
    print("  SUMMARY: ROUTING DECISION vs. PHYSICAL-LAYER OUTCOME")
    print("=" * 90)
    print(f"  {'Strategy':<22s} {'Path':<32s} "
          f"{'Tot km':>7s} {'Max km':>7s} {'p_bn':>8s} "
          f"{'#del':>5s} {'F̄':>8s}")
    print("  " + "-" * 88)
    for name in ("ShortestPathRouting", "MaxRateRouting"):
        r = results[name]
        print(f"  {name:<22s} {' → '.join(r['nodes']):<32s} "
              f"{r['total_km']:>7.0f} {r['bn_km']:>7.0f} "
              f"{r['p_bn']:>8.4f} {r['delivered']:>5d} {r['fbar']:>8.4f}")
    print("=" * 90)

    # ── Step 4: interpretation ────────────────────────────────────────
    print("\n  Interpretation:")
    print("  ─ ShortestPathRouting saves 15 km of total fibre but")
    print("    sacrifices 16× of per-attempt herald rate, because the")
    print("    single 100 km link dominates the end-to-end yield.")
    print("  ─ MaxRateRouting spends extra fibre to keep every link")
    print("    short, so end-to-end herald waiting time is far lower.")
    print("  ─ This is the fundamental repeater-network trade-off: the")
    print("    end-to-end rate depends on the *worst* link, not the")
    print("    sum of link lengths.\n")


# ═══════════════════════════════════════════════════════════════════════════
#  6.  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

async def main():
    await demo_convergent()
    await demo_divergent()


if __name__ == "__main__":
    asyncio.run(main())