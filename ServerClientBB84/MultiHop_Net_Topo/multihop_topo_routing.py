import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import asyncio
import time
import heapq
from typing import List, Dict, Tuple, Any, Optional
from abc import ABC, abstractmethod
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ServerClientBB84.MultiHop_Net_Topo.multihop_serverclient import (
    ProtocolWrapperServerClient, 
    MultiHopChain, 
    simulate_optimal_hops, 
    simulate_range_extension_protocols, 
    plot_optimal_hops, 
    plot_range_extension_protocols
)

# ==========================================
# MODULAR ROUTING STRATEGIES
# ==========================================

class RoutingStrategy(ABC):
    """Abstract base class for routing strategies in a Quantum Network."""
    @abstractmethod
    def find_path(self, graph: nx.Graph, source: str, target: str, wrapper: ProtocolWrapperServerClient) -> Tuple[List[str], float]:
        pass

class MaxRateRouting(RoutingStrategy):
    """
    Implements the 'Widest Path' algorithm to maximize the bottleneck rate.
    """
    def find_path(self, graph: nx.Graph, source: str, target: str, wrapper: ProtocolWrapperServerClient) -> Tuple[List[str], float]:
        # Pre-calculate rates for all edges in the graph
        rate_graph = nx.Graph()
        for u, v, d in graph.edges(data=True):
            rate, _ = wrapper.calculate_rate(d['distance'])
            rate_graph.add_edge(u, v, weight=rate)

        # Max-Min Path algorithm using a priority queue (Dijkstra-like)
        pq = [(-float('inf'), source, [source])]
        visited = set()
        best_rate = -1
        best_path = []

        while pq:
            (neg_current_rate, node, path) = heapq.heappop(pq)
            current_rate = -neg_current_rate

            if node == target:
                if current_rate > best_rate:
                    best_rate = current_rate
                    best_path = path
                continue

            if node in visited:
                continue
            visited.add(node)

            for neighbor in rate_graph.neighbors(node):
                edge_rate = rate_graph[node][neighbor]['weight']
                bottleneck = min(current_rate, edge_rate)
                heapq.heappush(pq, (-bottleneck, neighbor, path + [neighbor]))

        return best_path, best_rate

class ShortestPathRouting(RoutingStrategy):
    """
    Routes based on minimum physical distance (standard Dijkstra).
    """
    def find_path(self, graph: nx.Graph, source: str, target: str, wrapper: ProtocolWrapperServerClient) -> Tuple[List[str], float]:
        try:
            path = nx.shortest_path(graph, source=source, target=target, weight='distance')
            # Calculate the bottleneck rate for this specific path
            rates = [wrapper.calculate_rate(graph[path[i]][path[i+1]]['distance'])[0] for i in range(len(path)-1)]
            return path, min(rates) if rates else 0.0
        except nx.NetworkXNoPath:
            return [], 0.0

# ==========================================
# TOPOLOGY MANAGEMENT
# ==========================================

class QTopology:
    """
    Handles a custom topology of trusted nodes.
    """
    def __init__(self, nodes: List[str], edges: List[Tuple[str, str, float, float]]):
        self.graph = nx.Graph()
        self.graph.add_nodes_from(nodes)
        for u, v, dist, cap in edges:
            self.graph.add_edge(u, v, distance=dist, capacity=cap)

    def get_path_metrics(self, path: List[str], wrapper: ProtocolWrapperServerClient, chain_logic: MultiHopChain):
        link_rates, link_distances, link_capacities = [], [], []
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            edge_data = self.graph[u][v]
            dist, cap = edge_data['distance'], edge_data['capacity']
            rate_bps, _ = wrapper.calculate_rate(dist)
            link_rates.append(rate_bps)
            link_distances.append(dist)
            link_capacities.append(cap)

        eff_rate = chain_logic.calculate_chain_rate(link_rates, link_distances, link_capacities)
        return eff_rate, link_rates

    def route(self, source: str, target: str, wrapper: ProtocolWrapperServerClient, strategy: RoutingStrategy) -> Tuple[List[str], float]:
        """Executes a modular routing strategy to find the best path."""
        return strategy.find_path(self.graph, source, target, wrapper)

# ==========================================
# ANALYSIS AND VISUALIZATION
# ==========================================

def plot_topology(topo: QTopology, path: List[str] = None):
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(topo.graph)
    nx.draw(topo.graph, pos, with_labels=True, node_color='lightblue', 
            node_size=2000, font_weight='bold', edge_color='gray')
    if path:
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(topo.graph, pos, edgelist=path_edges, edge_color='red', width=3)
    edge_labels = nx.get_edge_attributes(topo.graph, 'distance')
    nx.draw_networkx_edge_labels(topo.graph, pos, edge_labels=edge_labels)
    plt.title("Quantum Network Topology (Edge weights = Distance km)")
    plt.show()

def run_topology_deep_analysis(
    source: str, 
    target: str, 
    topo: QTopology, 
    wrapper: ProtocolWrapperServerClient, 
    chain_logic: MultiHopChain,
    sim_config: Dict[str, Any],
    protocols: List[str] = ['polar', 'cascade', 'nr_ldpc_standard', 'ldpc_rateadaptive'],
    relay_counts: Optional[List[int]] = None,
    range_max_dist: float = 700.0,
    range_points: int = 20,
    chain_params: Optional[Dict] = None
):
    """
    Deep analysis with full user control over simulation parameters.
    """
    if relay_counts is None:
        relay_counts = np.linspace(1, 30, 20, dtype=int).tolist()
    if chain_params is None:
        chain_params = {'sifting_exchanges': 3, 'overhead_factor': 3, 'packet_size': 10000}

    print(f"\n--- Deep Analysis: {source} to {target} ---")
    
    # 1. Use MaxRateRouting by default for analysis
    router = MaxRateRouting()
    best_path, bottleneck_rate = topo.route(source, target, wrapper, router)
    
    path_dist = sum(topo.graph[best_path[i]][best_path[i+1]]['distance'] for i in range(len(best_path)-1))
    print(f"Routed Path: {' -> '.join(best_path)}")
    print(f"Total Path Distance: {path_dist:.2f} km")

    # 2. Optimal Hop Analysis
    opt_relays, opt_results = simulate_optimal_hops(
        protocols=protocols,
        pa_protocols=['toeplitz'],
        total_distance=path_dist,
        relay_counts=relay_counts,
        num_qubits=wrapper.num_qubits,
        link_capacity=1e9,
        sim_config=sim_config,
        chain_params=chain_params
    )
    plot_optimal_hops(opt_relays, opt_results, path_dist, 
                      title=f"Optimal Hop Analysis for Routed Path\n({source} to {target})")

    # 3. Range Extension Analysis
    scan_dists, range_results = simulate_range_extension_protocols(
        protocols=protocols,
        pa_protocols=['toeplitz'],
        max_distance=range_max_dist,
        num_points=range_points,
        comparison_relays=len(best_path)-2,
        num_qubits=wrapper.num_qubits,
        link_capacity=1e9,
        sim_config=sim_config
    )
    plot_range_extension_protocols(scan_dists, range_results, len(best_path)-2, sim_config)

def run_topology_example():
    # ==========================================
    # USER CONFIGURATION SECTION
    # ==========================================
    SIM_CONFIG = {
        'freq': 1e7, 'mu': 0.1, 'att_db_km': 0.2, 'det_eff': 0.8, 
        'dark_count': 0.001, 'protocol_params': {'u_fer_target': 0.01},
        'num_trials': 5
    }
    CHAIN_CONFIG = {'sifting_exchanges': 3, 'overhead_factor': 3, 'packet_size': 10000}
    
    # User defined lists for deep analysis
    ANALYSIS_PROTOCOLS = ['polar', 'cascade', 'nr_ldpc_standard', 'ldpc_rateadaptive']
    ANALYSIS_RELAY_COUNTS = np.linspace(1, 40, 20, dtype=int).tolist()
    
    # Topology definition
    nodes = ['Alice', 'R1', 'R2', 'R3', 'R4', 'Bob']
    edges = [
        ('Alice', 'R1', 50, 1e9), ('Alice', 'R2', 100, 1e9),
        ('R1', 'R2', 30, 1e9), ('R1', 'R3', 80, 1e9),
        ('R2', 'R3', 40, 1e9), ('R2', 'R4', 70, 1e9),
        ('R3', 'Bob', 100, 1e9), ('R4', 'Bob', 40, 1e9),
        ('R3', 'R4', 20, 1e9),
    ]
    # ==========================================

    wrapper = ProtocolWrapperServerClient(
        protocol='polar', pa_protocol='toeplitz', num_qubits=10000,
        freq=SIM_CONFIG['freq'], mu=SIM_CONFIG['mu'], 
        att_db_km=SIM_CONFIG['att_db_km'], detector_eff=SIM_CONFIG['det_eff'],
        dark_count=SIM_CONFIG['dark_count'], protocol_params=SIM_CONFIG['protocol_params']
    )
    wrapper.num_trials = SIM_CONFIG['num_trials']
    chain_logic = MultiHopChain(**CHAIN_CONFIG)
    topo = QTopology(nodes, edges)

    # Modular Routing: Choose strategy here (MaxRateRouting or ShortestPathRouting)
    routing_strategy = MaxRateRouting()
    best_path, bottleneck_rate = topo.route('Alice', 'Bob', wrapper, routing_strategy)
    eff_rate, link_rates = topo.get_path_metrics(best_path, wrapper, chain_logic)
    
    print(f"Routing Strategy: {routing_strategy.__class__.__name__}")
    print(f"Optimal Path: {' -> '.join(best_path)}")
    print(f"Effective End-to-End Rate: {eff_rate:.3e} bps")

    plot_topology(topo, path=best_path)
    
    # Run Deep Analysis with user-defined controls
    run_topology_deep_analysis(
        source='Alice', 
        target='Bob', 
        topo=topo, 
        wrapper=wrapper, 
        chain_logic=chain_logic, 
        sim_config=SIM_CONFIG,
        protocols=ANALYSIS_PROTOCOLS,
        relay_counts=ANALYSIS_RELAY_COUNTS,
        chain_params=CHAIN_CONFIG
    )

if __name__ == '__main__':
    run_topology_example()