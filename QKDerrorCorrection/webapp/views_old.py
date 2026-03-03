from django.shortcuts import render
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
from concurrent.futures import ThreadPoolExecutor

# Adjust path to include ServerClientBB84
# Current file is .../QKDerrorCorrection/webapp/views.py
# We want .../ServerClientBB84
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SERVER_CLIENT_PATH = os.path.join(BASE_DIR, 'ServerClientBB84')

if SERVER_CLIENT_PATH not in sys.path:
    sys.path.append(SERVER_CLIENT_PATH)

try:
    from bb84_server_client import AliceServer, BobClient, QuantumChannel, Detector, Eve, APIClient
except ImportError as e:
    print(f"Error importing BB84 modules: {e}")

# Create your views here.

def home(request):
    return render(request, "home.html", {})

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
                               cascade_passes=4, ldpc_rate="adaptive", eve_intercept=0.2, 
                               det_eff=0.8, det_dc=0.01):
    seed_base = 33333
    # Wiring
    channel = QuantumChannel("Fiber", length_km=distance_km, attenuation_db=attenuation_db, 
                             optical_error_rate=optical_error_rate, next_actor=None, seed=seed_base)
    alice = AliceServer("Alice", channel, num_qubits=num_qubits, verbose=False, seed=seed_base+1)
    api = APIClient(alice)
    
    # Pack protocol params
    p_params = {}
    if protocol == 'cascade':
        p_params = {'num_passes': cascade_passes}
    elif protocol.startswith('ldpc'):
        p_params = {'rate': ldpc_rate}

    bob = BobClient("Bob", api, protocol=protocol, seed=seed_base+2, verbose=False, protocol_params=p_params)
    
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

def run_simulation(request):
    context = {}
    if request.method == 'POST':
        # Get parameters
        try:
            # Main Params
            num_qubits = int(request.POST.get('num_qubits', 50000))
            distance_km = float(request.POST.get('distance_km', 50.0))
            
            # Channel / Physical
            channel_att = float(request.POST.get('channel_att', 0.2))
            
            # Sweep Params
            min_error = float(request.POST.get('min_error', 0.001))
            max_error = float(request.POST.get('max_error', 0.15))
            steps = int(request.POST.get('steps', 50))
            
            protocols = request.POST.getlist('protocols')
            if not protocols: protocols = ['cascade']
            
            # Component Params
            eve_intercept = float(request.POST.get('eve_intercept', 0.2))
            det_eff = float(request.POST.get('det_eff', 0.8))
            det_dc = float(request.POST.get('det_dc', 0.01))
            
            # Protocol Specific
            cascade_passes = int(request.POST.get('cascade_passes', 4))
            ldpc_rate_raw = request.POST.get('ldpc_rate', 'adaptive')
            # Try to convert ldpc_rate to float if it looks like a number
            try:
                ldpc_rate = float(ldpc_rate_raw)
            except ValueError:
                ldpc_rate = ldpc_rate_raw

        except ValueError:
            context['error'] = "Invalid input parameters"
            return render(request, "simulation.html", context)

        # Run Sweep
        error_rates = np.linspace(min_error, max_error, steps)
        
        # Prepare data structures
        data = {p: {"qber": [], "leakage": [], "efficiency": [], "uses": [], "key_rate": [], 
                    "scale_size": [], "scale_time": []} for p in protocols}

        async def run_sweep():
            # Experiment 1: QBER Sweep
            for p in protocols:
                for err in error_rates:
                    res = await _run_single_simulation(p, num_qubits, err, distance_km, channel_att,
                                                       cascade_passes, ldpc_rate, eve_intercept,
                                                       det_eff, det_dc)
                    if res.get('sifted_length', 0) > 0 and res.get('qber', 0) > 0:
                        eff = calculate_efficiency(res['revealed'], res['final_length'], res['qber'])
                        if eff:
                            data[p]["qber"].append(res['qber'] * 100) # %
                            # Leakage: fraction of reconciled key
                            leakage_frac = res['revealed'] / res['final_length']
                            data[p]["leakage"].append(leakage_frac * 100)
                            data[p]["efficiency"].append(eff)
                            data[p]["uses"].append(res['channel_uses'])
                            # Secure Key Rate = 1 - leakage(fraction) - h(QBER)
                            sec_rate = 1.0 - leakage_frac - binary_entropy(res['qber'])
                            data[p]["key_rate"].append(sec_rate)

            # Experiment 2: Computation Scalability (Fixed Error 2%)
            fixed_err_scale = 0.02
            # Use standard block sizes from script
            block_sizes = [1000, 2500, 5000, 10000, 20000] # Reduced max to 20k for web response time
            
            for p in protocols:
                for size in block_sizes:
                    res = await _run_single_simulation(p, size, fixed_err_scale, distance_km, channel_att,
                                                       cascade_passes, ldpc_rate, eve_intercept,
                                                       det_eff, det_dc)
                    if res.get('exec_time') is not None:
                        data[p]["scale_size"].append(size)
                        data[p]["scale_time"].append(res['exec_time'])

        asyncio.run(run_sweep())

        # Generate Graphs
        graphs = []
        
        # Helper to plot
        def plot_metric(metric_key, ylabel, title, log_scale=False, y_limit=None, x_key="qber", xlabel='Measured QBER (%)'):
            plt.figure(figsize=(10, 6))
            has_data = False
            for p, d in data.items():
                if d.get(x_key) and d.get(metric_key):
                    has_data = True
                    # Sort
                    sorted_pairs = sorted(zip(d[x_key], d[metric_key]))
                    xs, ys = zip(*sorted_pairs)
                    marker = 'o-' if p == 'cascade' else 's-'
                    # Basic color assignment
                    color = 'blue' if 'cascade' in p else 'red'
                    plt.plot(xs, ys, marker, color=color, label=p.upper())
            
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
            if metric_key == "key_rate":
                 plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            return img_str

        # Graph 1-3 & 5 (QBER Sweep)
        g1 = plot_metric("leakage", "Leakage (% of Key)", "Key Leakage vs QBER")
        if g1: graphs.append(g1)
        
        g2 = plot_metric("efficiency", "Efficiency (>1.0)", "Efficiency vs QBER (Shannon Limit=1.0)", y_limit=1.0)
        if g2: graphs.append(g2)
        
        g3 = plot_metric("uses", "Channel Uses", "Latency vs QBER", log_scale=True)
        if g3: graphs.append(g3)
        
        g5 = plot_metric("key_rate", "Final Key Rate (fraction)", "Secure Key Rate After Privacy Amplification")
        if g5: graphs.append(g5)
        
        # Graph 4 (Scalability)
        g4 = plot_metric("scale_time", "Execution Time (s)", "Computation Scalability (Block Size)", 
                         x_key="scale_size", xlabel="Block Size (Bits)")
        if g4: graphs.append(g4)

        context['graphs'] = graphs
        context['params'] = request.POST

    return render(request, "simulation.html", context)

