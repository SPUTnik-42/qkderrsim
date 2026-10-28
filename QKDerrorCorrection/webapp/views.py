from django.shortcuts import render
from django.http import StreamingHttpResponse
import sys
import os
import asyncio
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import math
import json
import networkx as nx
from concurrent.futures import ThreadPoolExecutor
from django.conf import settings

# Adjust path to include ServerClientBB84 and the workspace root for multihop utilities
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SERVER_CLIENT_PATH = os.path.join(BASE_DIR, 'ServerClientBB84')
WORKSPACE_ROOT = BASE_DIR

if SERVER_CLIENT_PATH not in sys.path:
    sys.path.append(SERVER_CLIENT_PATH)
if WORKSPACE_ROOT not in sys.path:
    sys.path.append(WORKSPACE_ROOT)

# Removed try-except block to expose import errors directly or debug them
# If this fails, check if bb84_server_client.py has syntax errors or missing dependencies
from ServerClientBB84.bb84_server_client import AliceServer, BobClient, QuantumChannel, Detector, Eve, APIClient
from multihop_serverclient import build_multihop_serverclient_dashboard
from ServerClientBB84.MultiHop_Net_Topo.multihop_topo_routing import MaxRateRouting, QTopology, ShortestPathRouting
from ServerClientBB84.MultiHop_Net_Topo.multihop_serverclient import (
    MultiHopChain,
    ProtocolWrapperServerClient,
    create_optimal_hops_figure,
    create_range_extension_protocols_figure,
    simulate_optimal_hops,
    simulate_range_extension_protocol_pairs,
    _figure_to_base64,
)

# Create your views here.

DEFAULT_NODES = ['Alice', 'R1', 'R2', 'R3', 'R4', 'Bob']

DEFAULT_EDGES = [
    {'source': 'Alice', 'target': 'R1', 'distance': 50},
    {'source': 'Alice', 'target': 'R2', 'distance': 100},
    {'source': 'R1',    'target': 'R2', 'distance': 30},
    {'source': 'R1',    'target': 'R3', 'distance': 80},
    {'source': 'R2',    'target': 'R3', 'distance': 40},
    {'source': 'R2',    'target': 'R4', 'distance': 70},
    {'source': 'R3',    'target': 'Bob','distance': 100},
    {'source': 'R4',    'target': 'Bob','distance': 40},
    {'source': 'R3',    'target': 'R4', 'distance': 20},
]

DEFAULT_NODE_COORDS = {
    'Alice': {'lat': 28.5500, 'lng': 77.0500},
    'R1':    {'lat': 28.6000, 'lng': 77.1300},
    'R2':    {'lat': 28.6000, 'lng': 77.2800},
    'R3':    {'lat': 28.6500, 'lng': 77.2000},
    'R4':    {'lat': 28.6500, 'lng': 77.3200},
    'Bob':   {'lat': 28.6800, 'lng': 77.4000},
}

def home(request):
    return render(request, "home.html", {})


def _normalise_topology_data(raw_topology_data):
    """
    Returns (nodes, edges, node_coords).

    Backwards compatible: string node lists still work.
    New shape: nodes = [{'id': 'Alice', 'lat': .., 'lng': ..}, ...]
    """
    if not raw_topology_data:
        return list(DEFAULT_NODES), [dict(e) for e in DEFAULT_EDGES], dict(DEFAULT_NODE_COORDS)

    try:
        parsed = json.loads(raw_topology_data)
        raw_nodes = parsed.get('nodes', [])
        edges = parsed.get('edges', [])
    except (TypeError, ValueError, AttributeError):
        return list(DEFAULT_NODES), [dict(e) for e in DEFAULT_EDGES], dict(DEFAULT_NODE_COORDS)

    clean_nodes, coords = [], {}
    for node in raw_nodes:
        if isinstance(node, str):
            nid = node.strip()
        elif isinstance(node, dict):
            nid = str(node.get('id', '')).strip()
        else:
            continue
        if nid and nid not in clean_nodes:
            clean_nodes.append(nid)

    clean_edges = []
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        source = str(edge.get('source', '')).strip()
        target = str(edge.get('target', '')).strip()
        if not source or not target or source == target:
            continue
        try:
            distance = max(float(edge.get('distance', 100)), 0.1)
        except (TypeError, ValueError):
            continue
        clean_edges.append({'source': source, 'target': target,
                            'distance': distance})
        for node in (source, target):
            if node not in clean_nodes:
                clean_nodes.append(node)

    if 'Alice' not in clean_nodes:
        clean_nodes.insert(0, 'Alice')
    if 'Bob' not in clean_nodes:
        clean_nodes.append('Bob')

    if not clean_edges:
        clean_edges = [dict(e) for e in DEFAULT_EDGES]

    for idx, nid in enumerate(clean_nodes):
        if nid not in coords:
            if nid in DEFAULT_NODE_COORDS:
                coords[nid] = dict(DEFAULT_NODE_COORDS[nid])
            else:
                angle = (idx / max(len(clean_nodes), 1)) * 2 * math.pi
                coords[nid] = {'lat': 28.6139 + 0.02 * math.cos(angle),
                               'lng': 77.2090 + 0.02 * math.sin(angle)}

    return clean_nodes, clean_edges, coords


def _topology_figure_base64(nodes, edges, path, node_coords=None):
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from((edge['source'], edge['target'], edge) for edge in edges)
    figure, axis = plt.subplots(figsize=(11, 7))
    if node_coords and all(n in node_coords for n in graph.nodes()):
        positions = {n: (node_coords[n]['lng'], node_coords[n]['lat']) for n in graph.nodes()}
    else:
        positions = nx.spring_layout(graph, seed=42)
    path_edges = set(zip(path, path[1:])) | set(zip(path[1:], path))
    edge_colors = ['#dc2626' if (u, v) in path_edges else '#94a3b8' for u, v in graph.edges()]
    nx.draw_networkx_edges(graph, positions, edge_color=edge_colors, width=[3 if color == '#dc2626' else 1.5 for color in edge_colors], ax=axis)
    nx.draw_networkx_nodes(graph, positions, node_color=['#bbf7d0' if node == 'Alice' else '#fecaca' if node == 'Bob' else '#bfdbfe' for node in graph.nodes()], node_size=1800, edgecolors='#334155', ax=axis)
    nx.draw_networkx_labels(graph, positions, font_weight='bold', ax=axis)
    nx.draw_networkx_edge_labels(graph, positions, edge_labels={(u, v): f"{data['distance']} km" for u, v, data in graph.edges(data=True)}, ax=axis)
    axis.set_title('Optimal Alice-Bob Path (red)')
    axis.axis('off')
    image = _figure_to_base64(figure)
    plt.close(figure)
    return image


def multihop_topology(request):
    default_form = {
        'protocols': ['polar', 'cascade', 'nr_ldpc_standard'],
        'pa_protocols': ['toeplitz'], 'num_qubits': 10000, 'freq': 1e7, 'mu': 0.1,
        'att_db_km': 0.2, 'det_eff': 0.8, 'dark_count': 0.001,
        'sifting_exchanges': 3, 'overhead_factor': 3, 'packet_size': 10000,
        'relay_sweep_max': 40, 'relay_sweep_points': 20, 'range_max_distance': 700,
        'range_points': 20,
        'routing_strategy': 'max_rate', 'num_trials': 5, 'base_seed': '',
        'cascade_passes': 4, 'winnow_passes': 4, 'ldpc_rate': None,
        'polar_u_fer': 0.01, 'polar_c': 0.5,
    }
    nodes, edges, node_coords = _normalise_topology_data(request.POST.get('topology_data'))
    context = {
        'submitted': False, 
        'nodes': json.dumps(nodes), 
        'edges': json.dumps(edges), 
        'node_coords': json.dumps(node_coords),
        'form_data': default_form, 
        'error': None
    }

    if request.method == 'POST':
        try:
            print('[multihop_topology] POST received')
            protocols = request.POST.getlist('ec_protocol') or default_form['protocols']
            pa_protocols = request.POST.getlist('pa_protocol') or default_form['pa_protocols']
            form_data = {key: request.POST.get(key, value) for key, value in default_form.items() if not isinstance(value, list)}
            form_data.update({
                'protocols': protocols, 'pa_protocols': pa_protocols,
                'num_qubits': int(request.POST.get('num_qubits', default_form['num_qubits'])),
                'freq': float(request.POST.get('freq', default_form['freq'])),
                'mu': float(request.POST.get('mu', default_form['mu'])),
                'att_db_km': float(request.POST.get('att_db_km', default_form['att_db_km'])),
                'det_eff': float(request.POST.get('det_eff', default_form['det_eff'])),
                'dark_count': float(request.POST.get('dark_count', default_form['dark_count'])),
                'sifting_exchanges': int(request.POST.get('sifting_exchanges', default_form['sifting_exchanges'])),
                'overhead_factor': int(request.POST.get('overhead_factor', default_form['overhead_factor'])),
                'packet_size': int(request.POST.get('packet_size', default_form['packet_size'])),
                'relay_sweep_max': max(1, int(request.POST.get('relay_sweep_max', default_form['relay_sweep_max']))),
                'relay_sweep_points': max(2, int(request.POST.get('relay_sweep_points', default_form['relay_sweep_points']))),
                'range_max_distance': max(10, float(request.POST.get('range_max_distance', default_form['range_max_distance']))),
                'range_points': max(2, int(request.POST.get('range_points', default_form['range_points']))),
                'routing_strategy': request.POST.get('routing_strategy', default_form['routing_strategy']),
                'num_trials': max(1, int(request.POST.get('num_trials', default_form['num_trials']))),
                'base_seed': request.POST.get('base_seed', default_form['base_seed']).strip(),
                'cascade_passes': max(1, int(request.POST.get('cascade_passes', default_form['cascade_passes']))),
                'winnow_passes': max(1, int(request.POST.get('winnow_passes', default_form['winnow_passes']))),
                'ldpc_rate': (
                    float(request.POST.get('ldpc_rate'))
                    if request.POST.get('ldpc_rate', '').strip()
                    else None
                ),
                'polar_u_fer': float(request.POST.get('polar_u_fer', default_form['polar_u_fer'])),
                'polar_c': float(request.POST.get('polar_c', default_form['polar_c'])),
            })
            topology = QTopology(nodes, [(edge['source'], edge['target'], edge['distance']) for edge in edges])
            primary_protocol = protocols[0]
            sim_config = {'freq': form_data['freq'], 'mu': form_data['mu'], 'att_db_km': form_data['att_db_km'], 'det_eff': form_data['det_eff'], 'dark_count': form_data['dark_count'], 'num_trials': form_data['num_trials'], 'protocol_params': {}}
            if form_data['base_seed']:
                sim_config['base_seed'] = int(form_data['base_seed'])
            if primary_protocol == 'polar':
                sim_config['protocol_params'] = {'u_fer_target': form_data['polar_u_fer'], 'c': form_data['polar_c']}
            elif primary_protocol in {'cascade', 'winnow'}:
                sim_config['protocol_params'] = {'num_passes': form_data['cascade_passes'] if primary_protocol == 'cascade' else form_data['winnow_passes']}
            elif primary_protocol == 'nr_ldpc_standard':
                sim_config['protocol_params'] = {'rate': form_data['ldpc_rate']}
            wrapper = ProtocolWrapperServerClient(protocol=primary_protocol, pa_protocol=pa_protocols[0], num_qubits=form_data['num_qubits'], freq=form_data['freq'], mu=form_data['mu'], att_db_km=form_data['att_db_km'], detector_eff=form_data['det_eff'], dark_count=form_data['dark_count'], protocol_params=sim_config['protocol_params'])
            routing_class = ShortestPathRouting if form_data['routing_strategy'] == 'shortest_path' else MaxRateRouting
            path, bottleneck_rate = topology.route('Alice', 'Bob', wrapper, routing_class())
            if not path:
                raise ValueError('No path connects Alice to Bob.')
            chain = MultiHopChain(form_data['sifting_exchanges'], form_data['overhead_factor'], form_data['packet_size'])
            effective_rate, link_rates = topology.get_path_metrics(path, wrapper, chain)
            path_edges = [{'distance': topology.graph[path[index]][path[index + 1]]['distance']} for index in range(len(path) - 1)]
            relay_counts = np.unique(np.linspace(1, form_data['relay_sweep_max'], form_data['relay_sweep_points'], dtype=int)).tolist()
            hop_x, hop_results = simulate_optimal_hops(protocols=protocols, pa_protocols=pa_protocols, total_distance=sum(edge['distance'] for edge in path_edges), relay_counts=relay_counts, num_qubits=form_data['num_qubits'], sim_config=sim_config, chain_params={'sifting_exchanges': form_data['sifting_exchanges'], 'overhead_factor': form_data['overhead_factor'], 'packet_size': form_data['packet_size']})
            range_x, range_results, range_relays = simulate_range_extension_protocol_pairs(path_edges, protocols=protocols, pa_protocols=pa_protocols, max_distance=form_data['range_max_distance'], num_points=form_data['range_points'], num_qubits=form_data['num_qubits'], sim_config=sim_config, chain_params={'sifting_exchanges': form_data['sifting_exchanges'], 'overhead_factor': form_data['overhead_factor'], 'packet_size': form_data['packet_size']})
            hop_figure = create_optimal_hops_figure(hop_x, hop_results, sum(edge['distance'] for edge in path_edges), title='Optimal Hop Analysis for Selected Path')
            range_figure = create_range_extension_protocols_figure(range_x, range_results, range_relays)
            context.update({
                'submitted': True, 
                'nodes': json.dumps(nodes), 
                'edges': json.dumps(edges), 
                'node_coords': json.dumps(node_coords),
                'form_data': form_data, 
                'path': path, 
                'bottleneck_rate': bottleneck_rate, 
                'effective_rate': effective_rate, 
                'total_distance': sum(edge['distance'] for edge in path_edges), 
                'plot_topology': _topology_figure_base64(nodes, edges, path, node_coords=node_coords), 
                'plot_optimal_hops': _figure_to_base64(hop_figure), 
                'plot_range_extension': _figure_to_base64(range_figure)
            })
            plt.close(hop_figure)
            plt.close(range_figure)
        except Exception as exc:
            print(f'[multihop_topology] Simulation failed: {exc}')
            context['error'] = f'Topology simulation failed: {exc}'
            context['node_coords'] = json.dumps(node_coords)

    return render(request, 'multihop_topo.html', context)


def _normalise_chain_data(raw_chain_data):
    defaults = [{"distance": 500}]
    if not raw_chain_data:
        return defaults

    try:
        parsed = json.loads(raw_chain_data)
    except (TypeError, ValueError):
        return defaults

    if not isinstance(parsed, list) or not parsed:
        return defaults

    chain = []
    for idx, segment in enumerate(parsed):
        if not isinstance(segment, dict):
            continue
        distance = float(segment.get('distance', 500 if idx == 0 else 100))
        chain.append({
            'distance': max(distance, 0.1),
        })

    return chain if chain else defaults


def _selected_protocols(post_data, field_name, fallback_values):
    values = [value for value in post_data.getlist(field_name) if value]
    return values if values else list(fallback_values)


def multihop_simulator(request):
    default_form = {
        'qkd_protocol': 'bb84',
        'protocol': 'cascade',
        'protocols': ['cascade'],
        'ec_protocols': ['cascade'],
        'pa_protocol': 'toeplitz',
        'pa_protocols': ['toeplitz'],
        'channel_mode': 'fiber',
        'freq': 1e7,
        'num_qubits': 10000,
        'mu': 0.1,
        'att_db_km': 0.2,
        'det_eff': 0.8,
        'dark_count': 0.01,
        'total_distance': 500,
        'sifting_exchanges': 3,
        'overhead_factor': 3,
        'packet_size': 10000,
        'relay_counts': '0',
        'relay_sweep_max': 40,
        'relay_sweep_points': 20,
        'range_relays': 0,
    }
    chain_data = _normalise_chain_data(request.POST.get('chain_data'))
    submitted = False
    effective_rate = 0.0
    total_distance = sum(float(seg.get('distance', 0.0)) for seg in chain_data)
    plot_optimal_hops = ''
    plot_range_extension = ''
    error = None

    gui_total_distance = total_distance
    gui_relay_count = max(0, len(chain_data) - 1)
    form_data = default_form.copy()

    if request.method == 'POST':
        selected_ec_protocols = _selected_protocols(
            request.POST,
            'ec_protocol',
            request.POST.getlist('protocol') or [request.POST.get('protocol', default_form['protocol'])],
        )
        primary_protocol = selected_ec_protocols[0] if selected_ec_protocols else default_form['protocol']

        selected_pa_protocols = _selected_protocols(
            request.POST,
            'pa_protocol',
            [request.POST.get('pa_protocol', default_form['pa_protocol'])],
        )
        primary_pa_protocol = selected_pa_protocols[0] if selected_pa_protocols else default_form['pa_protocol']

        relay_sweep_max = max(1, int(request.POST.get('relay_sweep_max', default_form['relay_sweep_max'])))
        relay_sweep_points = max(2, int(request.POST.get('relay_sweep_points', default_form['relay_sweep_points'])))

        gui_total_distance = float(
            request.POST.get('summary_total_distance', request.POST.get('total_distance', total_distance))
        )
        gui_relay_count = max(
            0,
            int(request.POST.get('summary_relay_count', request.POST.get('relay_counts', max(0, len(chain_data) - 1))))
        )

        form_data = {
            'qkd_protocol': 'bb84',
            'protocol': primary_protocol,
            'protocols': selected_ec_protocols if selected_ec_protocols else default_form['protocols'],
            'ec_protocols': selected_ec_protocols if selected_ec_protocols else default_form['ec_protocols'],
            'pa_protocol': primary_pa_protocol,
            'pa_protocols': selected_pa_protocols if selected_pa_protocols else default_form['pa_protocols'],
            'channel_mode': request.POST.get('channel_mode', default_form['channel_mode']),
            'freq': float(request.POST.get('freq', default_form['freq'])),
            'num_qubits': int(request.POST.get('num_qubits', default_form['num_qubits'])),
            'mu': float(request.POST.get('mu', default_form['mu'])),
            'att_db_km': float(request.POST.get('att_db_km', default_form['att_db_km'])),
            'det_eff': float(request.POST.get('det_eff', default_form['det_eff'])),
            'dark_count': float(request.POST.get('dark_count', default_form['dark_count'])),
            'total_distance': float(request.POST.get('total_distance', total_distance)),
            'sifting_exchanges': int(request.POST.get('sifting_exchanges', default_form['sifting_exchanges'])),
            'overhead_factor': int(request.POST.get('overhead_factor', default_form['overhead_factor'])),
            'packet_size': int(request.POST.get('packet_size', default_form['packet_size'])),
            'relay_counts': request.POST.get('relay_counts', default_form['relay_counts']),
            'relay_sweep_max': relay_sweep_max,
            'relay_sweep_points': relay_sweep_points,
            'range_relays': gui_relay_count,
            'summary_total_distance': gui_total_distance,
            'summary_relay_count': gui_relay_count,
        }
        chain_data = _normalise_chain_data(request.POST.get('chain_data'))
        total_distance = sum(float(seg.get('distance', 0.0)) for seg in chain_data)
        form_data['total_distance'] = total_distance
        gui_total_distance = float(request.POST.get('summary_total_distance', total_distance))
        gui_relay_count = max(
            0,
            int(request.POST.get('summary_relay_count', max(0, len(chain_data) - 1)))
        )

        sim_config = {
            'protocol': primary_protocol,
            'pa_protocol': primary_pa_protocol,
            'freq': form_data['freq'],
            'mu': form_data['mu'],
            'att_db_km': form_data['att_db_km'],
            'det_eff': form_data['det_eff'],
            'dark_count': form_data['dark_count'],
            'num_trials': 3,
            'protocol_params': {},
        }
        if primary_protocol == 'polar':
            sim_config['protocol_params'] = {'u_fer_target': 0.01}
        elif primary_protocol in {'cascade', 'winnow'}:
            sim_config['protocol_params'] = {'num_passes': 4}
        elif primary_protocol == 'nr_ldpc_standard':
            sim_config['protocol_params'] = {'rate': 0.333}

        chain_params = {
            'sifting_exchanges': form_data['sifting_exchanges'],
            'overhead_factor': form_data['overhead_factor'],
            'packet_size': form_data['packet_size'],
        }

        try:
            dashboard = build_multihop_serverclient_dashboard(
                chain_data=chain_data,
                selected_ec_protocols=selected_ec_protocols,
                selected_pa_protocols=selected_pa_protocols,
                num_qubits=form_data['num_qubits'],
                sim_config=sim_config,
                chain_params=chain_params,
                relay_sweep_max=relay_sweep_max,
                relay_sweep_points=relay_sweep_points,
                max_distance=max(gui_total_distance, 200.0),
            )
            effective_rate = float(dashboard.get('effective_rate', 0.0))
            total_distance = float(dashboard.get('total_distance', total_distance))
            submitted = True
            plot_optimal_hops = dashboard.get('plot_optimal_hops', '')
            plot_range_extension = dashboard.get('plot_range_extension', '')

        except Exception as exc:
            error = f'Simulation failed: {exc}'
            submitted = False
            effective_rate = 0.0

    return render(request, 'multihop_simulator.html', {
        'submitted': submitted,
        'effective_rate': effective_rate,
        'total_distance': total_distance,
        'plot_optimal_hops': plot_optimal_hops,
        'plot_range_extension': plot_range_extension,
        'chain_data': json.dumps(chain_data),
        'segments_count': len(chain_data),
        'form_data': form_data,
        'error': error,
    })


def binary_entropy(p):
    """Calculates binary entropy H(p)."""
    if p <= 0 or p >= 1: return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

def calculate_efficiency(n_revealed, k_sifted, qber):
    if k_sifted == 0 or qber <= 0 or qber >= 1:
        return None
    h_eps = binary_entropy(qber)
    if h_eps == 0: return None
    return n_revealed / (k_sifted * h_eps)

async def _run_single_simulation(protocol, num_qubits, optical_error_rate, distance_km, attenuation_db,
                               cascade_passes=4, winnow_passes=4, ldpc_rate=None,
                               polar_u_fer=0.01, polar_c=0.5,
                               eve_intercept=0.2, det_eff=0.8, det_dc=0.01,
                               ldpc_target_fer=0.05, ldpc_num_iter=30, ldpc_max_k=4224,
                               ldpc_sweep_factor=1.07, ldpc_max_sweep_iter=12,
                               ldpc_hash_bits=12):
    print(f"Running {protocol.upper()} | Qubits: {num_qubits} | Error: {optical_error_rate:.4f} | Dist: {distance_km}km")
    seed_base = 33333
    # Wiring
    channel = QuantumChannel("Fiber", length_km=distance_km, attenuation_db=attenuation_db, 
                             optical_error_rate=optical_error_rate, next_actor=None, seed=seed_base)
    alice = AliceServer("Alice", channel, num_qubits=num_qubits, verbose=True, seed=seed_base+1)
    api = APIClient(alice)
    
    # Pack protocol params
    p_params = {}
    if protocol == 'cascade':
        p_params = {'num_passes': cascade_passes}
    elif protocol == 'winnow':
        p_params = {'num_passes': winnow_passes}
    elif protocol == 'polar':
        p_params = {'u_fer_target': polar_u_fer, 'c': polar_c}
    elif protocol in {'nr_ldpc_standard', 'nr_ldpc_std', 'ldpc_5g_std'}:
        p_params = {
            'rate': ldpc_rate,
            'target_fer': ldpc_target_fer,
            'num_iter': ldpc_num_iter,
            'max_k': ldpc_max_k,
            'sweep_factor': ldpc_sweep_factor,
            'max_sweep_iter': ldpc_max_sweep_iter,
            'hash_bits': ldpc_hash_bits,
        }

    bob = BobClient("Bob", api, protocol=protocol, seed=seed_base+2, verbose=True, protocol_params=p_params)
    
    detector = Detector("Detector", efficiency=det_eff, dark_count_prob=det_dc, parent_bob=bob, seed=seed_base+3)
    
    # Eve
    eve = Eve("Eve", next_actor=detector, intercept_rate=eve_intercept, seed=seed_base+5)
    channel.next_actor = eve
    
    actors = [alice, channel, eve, detector, bob]
    tasks = [asyncio.create_task(a.start()) for a in actors]
    
    results = {}
    try:
        await alice.run_quantum_transmission()
        await asyncio.sleep(0.1) # Wait for photons
        results = await bob.run_classical_post_processing(alice.num_qubits)
    except Exception as e:
        print(f"Sim Error: {e}")
    finally:
        for a in actors: a._running = False
        # Cancel tasks
        for t in tasks: t.cancel()
    
    return results

class OutputCapture(io.StringIO):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue
        
    def write(self, s):
        if s:
            # Escape for JS string
            safe_s = s.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '')
            self.queue.put_nowait(safe_s)
        # We don't actually need to store the string in memory if we are just streaming it.
        # super().write(s) would keep it in RAM forever.
        return len(s)

def run_simulation(request):
    defaults = {
        'num_qubits': 50000, 'steps': 50, 'min_error': 0.001, 'max_error': 0.15,
        'distance_km': 50, 'channel_att': 0.2, 'eve_intercept': 0.2,
        'det_eff': 0.8, 'det_dc': 0.01, 'cascade_passes': 4, 'winnow_passes': 4,
        'ldpc_rate': None, 'ldpc_target_fer': 0.05, 'ldpc_num_iter': 30,
        'ldpc_max_k': 4224, 'ldpc_sweep_factor': 1.07,
        'ldpc_max_sweep_iter': 12, 'ldpc_hash_bits': 32,
        'polar_u_fer': 0.01, 'polar_c': 0.5,
        'protocols': ['cascade', 'nr_ldpc_standard', 'winnow', 'polar'],
    }

    if request.method != 'POST':
        return render(request, "simulation.html", {'params': defaults})

    # Parameters
    try:
        # Main Params
        num_qubits = int(request.POST.get('num_qubits', 50000))
        distance_km = float(request.POST.get('distance_km', 50.0))
        channel_att = float(request.POST.get('channel_att', 0.2))
        
        min_error = float(request.POST.get('min_error', 0.001))
        max_error = float(request.POST.get('max_error', 0.15))
        steps = int(request.POST.get('steps', 50))
        
        protocols = request.POST.getlist('protocols')
        if not protocols: protocols = ['cascade']
        
        eve_intercept = float(request.POST.get('eve_intercept', 0.2))
        det_eff = float(request.POST.get('det_eff', 0.8))
        det_dc = float(request.POST.get('det_dc', 0.01))
        
        cascade_passes = int(request.POST.get('cascade_passes', 4))
        winnow_passes = int(request.POST.get('winnow_passes', 4))
        polar_u_fer = float(request.POST.get('polar_u_fer', 0.01))
        polar_c = float(request.POST.get('polar_c', 0.5))
        ldpc_rate_raw = request.POST.get('ldpc_rate', '')
        ldpc_rate = (
            float(ldpc_rate_raw)
            if str(ldpc_rate_raw).strip()
            else None
        )
        ldpc_target_fer = float(request.POST.get('ldpc_target_fer', defaults['ldpc_target_fer']))
        ldpc_num_iter = int(request.POST.get('ldpc_num_iter', defaults['ldpc_num_iter']))
        ldpc_max_k = int(request.POST.get('ldpc_max_k', defaults['ldpc_max_k']))
        ldpc_sweep_factor = float(request.POST.get('ldpc_sweep_factor', defaults['ldpc_sweep_factor']))
        ldpc_max_sweep_iter = int(request.POST.get('ldpc_max_sweep_iter', defaults['ldpc_max_sweep_iter']))
        ldpc_hash_bits = int(request.POST.get('ldpc_hash_bits', defaults['ldpc_hash_bits']))

        if ldpc_rate is not None and not 0.0 < ldpc_rate <= 1.0:
            raise ValueError("LDPC rate must be between 0 and 1")
        if not 0.0 < ldpc_target_fer < 1.0:
            raise ValueError("LDPC target FER must be between 0 and 1")
        if ldpc_num_iter < 1 or ldpc_max_k < 24 or ldpc_max_sweep_iter < 1:
            raise ValueError("LDPC iteration and block-size parameters are invalid")
        if ldpc_hash_bits < 1:
            raise ValueError("LDPC hash size must be positive")

    except ValueError:
        return render(request, "simulation.html", {'error': "Invalid input parameters", 'params': request.POST})

    # Prepare data storage
    data = {p: {"qber": [], "leakage": [], "efficiency": [], "uses": [],
                "estimated_key_rate": [], "realized_qber": [], "realized_key_rate": [],
                "scale_size": [], "scale_time": [],
                "sifted_length": [], "final_length": [], "revealed": [], "errors_corrected": []} for p in protocols}


    # Helper function for yielding log messages as simple JS/HTML updates
    def stream_response_generator():
        # 1. Yield initial HTML structure
        initial_context = {'params': request.POST, 'streaming_mode': True}
        yield render(request, "simulation.html", initial_context).content

        import queue
        from queue import Empty
        
        # We need a Thread to run the simulation because run_until_complete blocks the thread.
        # But we need to switch stdout. 
        # Modifying sys.stdout is tricky in threaded environment if there are concurrent requests, but for this demo it is likely single user.
        # Alternatively, we can use a custom print function injected into the modules, but that's invasive.
        # Let's use a queue and a thread.
        
        log_queue = queue.Queue()
        
        # Wrapper to run simulation and capture logs
        def simulation_thread_target():
            # Capture stdout for this thread
            # NOTE: sys.stdout is global. Redirecting it affects ALL threads/requests. 
            # This is a known limitation. For a local demo, it's acceptable.
            # Ideally we'd replace print() with a logging call, but that requires editing all files.
            
            original_stdout = sys.stdout
            capturer = OutputCapture(log_queue)
            sys.stdout = capturer
            
            try:
                # We need a new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # --- The Simulation Logic (Copied from before) ---
                error_rates = np.linspace(min_error, max_error, steps)
                
                # Experiment 1
                print("Starting QBER Sweep...")
                for p in protocols:
                    for err in error_rates:
                        res = loop.run_until_complete(_run_single_simulation(p, num_qubits, err, distance_km, channel_att,
                                                            cascade_passes, winnow_passes, ldpc_rate,
                                                            polar_u_fer, polar_c, eve_intercept, det_eff, det_dc,
                                                            ldpc_target_fer, ldpc_num_iter, ldpc_max_k,
                                                            ldpc_sweep_factor, ldpc_max_sweep_iter, ldpc_hash_bits))
                        
                        if res.get('sifted_length', 0) > 0 and res.get('qber', 0) > 0:
                            eff = calculate_efficiency(res['revealed'], res['final_length'], res['qber'])
                            if eff:
                                data[p]["qber"].append(res['qber'] * 100)
                                leakage_frac = res['revealed'] / res['final_length']
                                data[p]["leakage"].append(leakage_frac * 100)
                                data[p]["efficiency"].append(eff)
                                data[p]["uses"].append(res['channel_uses'])
                                data[p]["sifted_length"].append(res['sifted_length'])
                                data[p]["final_length"].append(res['final_length'])
                                data[p]["revealed"].append(res['revealed'])
                                if 'errors_corrected' in res:
                                    data[p]["errors_corrected"].append(res['errors_corrected'])
                                # Match vis_comparison_ec_script.py:
                                # estimated SKF uses reconciliation efficiency, while
                                # realized SKF is the privacy-amplified length actually produced.
                                estimated_skf = 1.0 - (
                                    1.0 + eff
                                ) * binary_entropy(res['qber'])
                                data[p]["estimated_key_rate"].append(estimated_skf)
                                sifted_length = res['sifted_length']
                                pa_length = res.get('pa_length')
                                if pa_length is not None and sifted_length > 0:
                                    data[p]["realized_qber"].append(res['qber'] * 100)
                                    data[p]["realized_key_rate"].append(
                                        float(pa_length) / sifted_length
                                    )

                # Experiment 2
                print("\nStarting Scalability Tests...")
                fixed_err_scale = 0.02
                block_sizes = [1000, 2500, 5000, 10000, 20000]
                
                for p in protocols:
                    for size in block_sizes:
                        res = loop.run_until_complete(_run_single_simulation(p, size, fixed_err_scale, distance_km, channel_att,
                                                            cascade_passes, winnow_passes, ldpc_rate,
                                                            polar_u_fer, polar_c, eve_intercept, det_eff, det_dc,
                                                            ldpc_target_fer, ldpc_num_iter, ldpc_max_k,
                                                            ldpc_sweep_factor, ldpc_max_sweep_iter, ldpc_hash_bits))
                        if res.get('exec_time') is not None:
                            data[p]["scale_size"].append(size)
                            data[p]["scale_time"].append(res['exec_time'])
                            
                loop.close()
                print("\nSimulation Complete. Generating Graphs...")
                
            except Exception as e:
                print(f"Simulation Error: {e}")
            finally:
                sys.stdout = original_stdout
                # Signal done
                log_queue.put(None) 

        # Start thread
        import threading
        import time 
        t = threading.Thread(target=simulation_thread_target)
        t.start()


        # Main generator loop: consume queue and yield to browser
        while True:
            try:
                # Batch processing: get all available items from queue at once
                # to reduce HTTP streaming overhead and string concatenation cost
                msgs = []
                
                # Block for the first item (with timeout to check if thread is alive)
                try:
                    first_msg = log_queue.get(timeout=0.1)
                    if first_msg is None: # Sentinel
                        break
                    msgs.append(first_msg)
                except Empty:
                    if not t.is_alive():
                        break
                    yield b'' # Keep alive
                    continue

                # Consuming remaining items without blocking
                while not log_queue.empty():
                    try:
                        m = log_queue.get_nowait()
                        if m is None:
                            break
                        msgs.append(m)
                    except Empty:
                        break
                
                if not msgs:
                    continue

                # Combine into fewer script tags
                combined_log = "".join(msgs)
                
                # Careful with escaping. "combined_log" already has escaped backslashes and newlines from OutputCapture.
                # But since we are putting it into a Python f-string that becomes a JS string literal...
                # OutputCapture turned \n into \\n (two chars: backslash, n).
                # If we put that into fString:
                #    JS: document.createTextNode("... combined_log ...")
                # If combined_log has `\\n`, JS sees `\n` (newline char) because backslash escapes n.
                # If original had `\`, OutputCapture made it `\\`. F-string passes `\\`. JS sees `\` (escaped backslash).
                # It seems correct.
                
                yield f'<script>document.getElementById("sim-log").appendChild(document.createTextNode("{combined_log}"));</script>'.encode()
                
                # Use the new global function if available (auto-scroll behavior)
                # But rate limit scroll calls slightly to avoid browser thrashing if we are streaming gigabytes
                # Actually, plain JS is fast enough. The browser renders on frames.
                yield b'<script>if(window.updateTerminalScroll) window.updateTerminalScroll();</script>'
                
            except Exception:
                break 

        # 3. Generate Graphs (Same as before)
        graphs = []
        def plot_metric(metric_key, ylabel, title, log_scale=False, y_limit=None, x_key="qber", xlabel='Measured QBER (%)'):
            plt.figure(figsize=(10, 6))
            has_data = False
            for p, d in data.items():
                if d.get(x_key) and d.get(metric_key):
                    has_data = True
                    sorted_pairs = sorted(zip(d[x_key], d[metric_key]))
                    xs, ys = zip(*sorted_pairs)
                    if 'cascade' in p:
                        marker, color = 'o-', 'blue'
                    elif 'winnow' in p:
                        marker, color = '^-', 'green'
                    elif p in {'nr_ldpc_standard', 'nr_ldpc_std', 'ldpc_5g_std'}:
                        marker, color = 's-', 'purple'
                    elif 'ldpc' in p:
                        marker, color = 'v-', 'red'
                    elif 'polar' in p:
                        marker, color = 'd-', 'orange'
                    else: # Fallback
                        marker, color = 'x-', 'black'
                    
                    plt.plot(xs, ys, marker, color=color, label=p.replace('_', ' ').upper())
            
            if not has_data:
                plt.close()
                return None
            
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
            plt.legend()
            plt.grid(True, alpha=0.3)
            if log_scale: plt.yscale('log')
            if y_limit: plt.axhline(y=y_limit, color='green', linestyle='--', label='Limit')
            if metric_key in {"key_rate", "estimated_key_rate", "realized_key_rate"}:
                 plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            return img_str

        # Same graph generation logic
        g1 = plot_metric("leakage", "Leakage (% of Key)", "Key Leakage vs QBER")
        if g1: graphs.append(g1)
        g2 = plot_metric("efficiency", "Efficiency (>1.0)", "Efficiency vs QBER (Shannon Limit=1.0)", y_limit=1.0)
        if g2: graphs.append(g2)
        g3 = plot_metric("uses", "Channel Uses", "Latency vs QBER", log_scale=True)
        if g3: graphs.append(g3)
        estimated_skf_graph = plot_metric(
            "estimated_key_rate",
            "Estimated Secret Key Fraction",
            "Estimated Secret Key Fraction vs QBER",
        )
        if estimated_skf_graph:
            graphs.append(estimated_skf_graph)
        realized_skf_graph = plot_metric(
            "realized_key_rate",
            "Realized Secret Key Fraction",
            "Realized Secret Key Fraction vs QBER",
            x_key="realized_qber",
        )
        if realized_skf_graph:
            graphs.append(realized_skf_graph)
        g4 = plot_metric("scale_time", "Execution Time (s)", "Computation Scalability (Block Size)", x_key="scale_size", xlabel="Block Size (Bits)")
        if g4: graphs.append(g4)

        # Compute metrics panel HTML
        metrics_html = '<div class="metrics-panel" style="background: white; padding: 20px; border-radius: 8px; margin-bottom: 30px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); border: 1px solid #e2e8f0;">'
        metrics_html += '<h3 style="margin-top: 0; color: #2c5282;">Simulation Metrics</h3>'
        has_metrics = False
        
        all_sifted = []
        all_final = []
        for p, d in data.items():
            if d.get("qber"):
                all_sifted.extend(d["sifted_length"])
                all_final.extend(d["final_length"])
                
        if all_sifted and all_final:
            has_metrics = True
            avg_sifted_global = np.mean(all_sifted)
            avg_final_global = np.mean(all_final)
            
            metrics_html += '<div class="common-metrics" style="margin-bottom: 25px;">'
            metrics_html += '<h4 style="margin-bottom: 10px; color: #2b6cb0; border-bottom: 2px solid #bee3f8; padding-bottom: 5px;">Common Metrics</h4>'
            metrics_html += '<ul style="list-style-type: none; padding-left: 0; margin: 0; display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 10px;">'
            metrics_html += f'<li style="background: #e6fffa; padding: 10px; border-radius: 4px; border-left: 4px solid #319795;"><strong>Avg Sifted Key Length:</strong> {avg_sifted_global:.1f}</li>'
            metrics_html += f'<li style="background: #e6fffa; padding: 10px; border-radius: 4px; border-left: 4px solid #319795;"><strong>Avg Final Key Length:</strong> {avg_final_global:.1f}</li>'
            metrics_html += '</ul></div>'
        
        for p, d in data.items():
            if not d.get("qber"): continue
            has_metrics = True
            metrics_html += f'<div class="protocol-metrics" style="margin-bottom: 20px;">'
            metrics_html += f'<h4 style="margin-bottom: 10px; color: #4a5568; border-bottom: 1px solid #e2e8f0; padding-bottom: 5px;">{p.replace("_", " ").upper()}</h4>'
            
            avg_eff = np.mean(d["efficiency"])
            avg_revealed = np.mean(d["revealed"])
            
            metrics_html += '<ul style="list-style-type: none; padding-left: 0; margin: 0; display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 10px;">'
            metrics_html += f'<li style="background: #f7fafc; padding: 10px; border-radius: 4px;"><strong>Avg Reconciliation Efficiency:</strong> {avg_eff:.3f}</li>'
            metrics_html += f'<li style="background: #f7fafc; padding: 10px; border-radius: 4px;"><strong>Avg Total Bits Revealed:</strong> {avg_revealed:.1f}</li>'
            
            if d.get("errors_corrected"):
                avg_err_corr = np.mean(d["errors_corrected"])
                metrics_html += f'<li style="background: #f7fafc; padding: 10px; border-radius: 4px;"><strong>Avg Errors Corrected:</strong> {avg_err_corr:.1f}</li>'
            
            metrics_html += '</ul></div>'
            
        metrics_html += '</div>'
        
        if not has_metrics:
            metrics_html = ""
            
        metrics_html = metrics_html.replace('"', '\\"').replace('\n', '')

        # 4. Yield Graph HTML
        # Clear the "InProgress" placeholder first
        if graphs or has_metrics:
            yield b'<script>document.getElementById("graphs-container").innerHTML = "";</script>'
            yield b'''<script>document.getElementById("graphs-container").innerHTML =
                '<div class="graph-card" style="padding-bottom: 12px;"><p style="margin: 0; color: #4a5568;"><strong>Secret key fractions:</strong> estimated fraction uses the measured QBER and reconciliation efficiency; realized fraction uses the privacy-amplified key length returned by the protocol.</p></div>';</script>'''
            if metrics_html:
                yield f'<script>document.getElementById("metrics-container").innerHTML = "{metrics_html}";</script>'.encode()
            
            for g in graphs:
                # Use a unique ID for each graph to avoid conflicts if needed, but not necessary here
                # Warning: JS string escaping for base64 might be tricky if not careful, but base64 is safe.
                img_tag = f'<div class="graph-card"><img src="data:image/png;base64,{g}" alt="Simulation Graph"></div>'
                yield f'<script>document.getElementById("graphs-container").innerHTML += \'{img_tag}\';</script>'.encode()
        else:
            yield b'<script>document.getElementById("graphs-container").innerHTML = "<div style=\'text-align:center; padding: 20px;\'>No valid data generated to plot.</div>";</script>'

    return StreamingHttpResponse(stream_response_generator())
