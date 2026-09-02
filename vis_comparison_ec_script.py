import asyncio
import random
import sys
import os
import math
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np

# Setup proper imports
current_dir = os.getcwd()
target_dir = os.path.join(current_dir, 'ServerClientBB84')
if target_dir not in sys.path:
    sys.path.append(target_dir)

try:
    from bb84_server_client import AliceServer, BobClient, QuantumChannel, Detector, Eve, APIClient
except ImportError:
    sys.path.append(os.getcwd())
    from bb84_server_client import AliceServer, BobClient, QuantumChannel, Detector, Eve, APIClient

def binary_entropy(p):
    if p <= 0 or p >= 1: return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

def calculate_efficiency(n_revealed, k_sifted, qber):
    if k_sifted == 0 or qber <= 0 or qber >= 1: return None
    h_eps = binary_entropy(qber)
    if h_eps == 0: return None
    return n_revealed / (k_sifted * h_eps)

async def run_comparison_simulation(protocol="cascade", num_qubits=5000, optical_error_rate=0.01, verbose=False, seed_base=44444):
    channel = QuantumChannel("Fiber", length_km=50, attenuation_db=0.2, 
                             optical_error_rate=optical_error_rate, next_actor=None, seed=seed_base)
    
    alice = AliceServer("Alice", channel, num_qubits=num_qubits, verbose=verbose, seed=seed_base+1)
    api = APIClient(alice)
    
    bob = BobClient("Bob", api, protocol=protocol, seed=seed_base+2, verbose=verbose)
    detector = Detector("Detector", efficiency=0.8, dark_count_prob=0.01, parent_bob=bob, seed=seed_base+3)
    eve = Eve("Eve", next_actor=detector, intercept_rate=0.2, seed=seed_base+5)
    channel.next_actor = eve
    
    actors = [alice, channel, eve, detector, bob]
    tasks = [asyncio.create_task(a.start()) for a in actors]
    
    results = {}
    try:
        start_time = time.time()
        await alice.run_quantum_transmission()
        
        # Add tiny delay based on block size
        wait_time = 0.5 + (num_qubits * 0.0001)
        await asyncio.sleep(wait_time)
        
        res = await bob.run_classical_post_processing(num_qubits)
        if res:
            results.update(res)
        end_time = time.time()
        results["exec_time"] = end_time - start_time
        
    except Exception as e:
        if verbose:
            print(f"Error in {protocol} sim: {e}")
        results = {'sifted_length': 0}
    finally:
        for a in actors:
            await a.send(a, ("STOP",))
        await asyncio.gather(*tasks, return_exceptions=True)
        
    return results

async def run_multiple_simulations(protocol, num_qubits, optical_error_rate, runs, verbose=False):
    """Run simulations concurrently and compute the average."""
    # Use deterministic seeds starting from the notebook's default (44444)
    tasks = [
        run_comparison_simulation(protocol, num_qubits, optical_error_rate, verbose, seed_base=44444 + (i * 10)) 
        for i in range(runs)
    ]
    all_results = await asyncio.gather(*tasks)
    
    # Filter out failures
    valid_results = [r for r in all_results if r.get('sifted_length', 0) > 0 and r.get('qber', 0) > 0]
    
    if not valid_results:
        return None
        
    # Average the valid runs
    avg_res = {
        "qber": np.mean([r['qber'] for r in valid_results]),
        "revealed": np.mean([r['revealed'] for r in valid_results]),
        "final_length": np.mean([r['final_length'] for r in valid_results]),
        "channel_uses": np.mean([r['channel_uses'] for r in valid_results]),
        "exec_time": np.mean([r.get('exec_time', 0) for r in valid_results]),
        "sifted_length": np.mean([r['sifted_length'] for r in valid_results])
    }
    return avg_res


async def main():
    parser = argparse.ArgumentParser(description="BB84 Reconciliation Comparison Script")
    parser.add_argument("--runs", type=int, default=10, help="Number of Monte Carlo iterations to average per data point")
    parser.add_argument("--qubits", type=int, default=100000, help="Initial sequence size for QBER sweep")
    parser.add_argument("--outdir", type=str, default="figures/serverPlots_Finiteasym/", help="Directory to save the resulting plots")
    parser.add_argument("--protocols", nargs='+', default=["cascade", "nr_ldpc_std", "ldpc_rateadaptive", "winnow", "polar"], 
                        help="List of protocols to evaluate")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print(f"--- Configuration ---")
    print(f"Runs per point: {args.runs}")
    print(f"N Qubits for Sweep: {args.qubits}")
    print(f"Enabled protocols: {', '.join(args.protocols)}")
    print(f"Output Directory: {args.outdir}\n")

    # Style mapping
    styles = {
        "cascade": {'color': 'blue', 'marker': 'o-', 'label': 'Cascade'},
        "nr_ldpc_std": {'color': 'purple', 'marker': 's-', 'label': 'NR LDPC 5G (Std)'},
        "ldpc_rateadaptive": {'color': 'red', 'marker': 'v-', 'label': 'Rate-Adaptive LDPC'},
        "winnow": {'color': 'green', 'marker': '^-', 'label': 'Winnow'},
        "polar": {'color': 'orange', 'marker': 'd-', 'label': 'Polar Codes'}
    }

    # Data stores
    data_exp1 = {p: {"qber": [], "leakage": [], "efficiency": [], "uses": []} for p in args.protocols}
    
    # ---------------- EXPERIMENT 1 ----------------
    print("\n--- Starting Experiment 1: Impact of QBER ---")
    error_rates = np.linspace(0.001, 0.12, 25) 
    
    for err in error_rates:
        print(f"\n[Exp 1] Testing Optical Error: {err:.4f}")
        for protocol in args.protocols:
            avg_res = await run_multiple_simulations(protocol, args.qubits, err, args.runs)
            
            if avg_res and avg_res.get('sifted_length', 0) > 0 and avg_res.get('qber', 0) > 0:
                eff = calculate_efficiency(avg_res['revealed'], avg_res['final_length'], avg_res['qber'])
                if eff:
                    data_exp1[protocol]["qber"].append(avg_res['qber'] * 100)
                    data_exp1[protocol]["leakage"].append((avg_res['revealed'] / avg_res['final_length']) * 100)
                    data_exp1[protocol]["efficiency"].append(eff)
                    data_exp1[protocol]["uses"].append(avg_res['channel_uses'])
            
            print(f"  > {protocol}: avg QBER={avg_res['qber']*100:.2f}%" if avg_res else f"  > {protocol}: Failed")

    # ---------------- EXPERIMENT 2 ----------------
    print("\n--- Starting Experiment 2: Computation Time vs Block Size ---")
    block_sizes = [1000, 2500, 5000, 10000, 20000]
    FIXED_ERR = 0.02
    data_exp2 = {p: [] for p in args.protocols}

    for size in block_sizes:
        print(f"\n[Exp 2] Testing Block Size: {size}")
        for protocol in args.protocols:
            avg_res = await run_multiple_simulations(protocol, size, FIXED_ERR, args.runs)
            exec_time = avg_res.get('exec_time', 0) if avg_res else 0
            data_exp2[protocol].append(exec_time)
            print(f"  > {protocol}: {exec_time:.4f} sec")


    # ---------------- PLOTTING EXPORT ----------------
    print("\n--- Generating Plots ---")
    
    # Plot 1: Leakage
    plt.figure(figsize=(10, 6))
    for p in args.protocols:
        if data_exp1[p]["qber"]:
            qber_arr = np.array(data_exp1[p]["qber"])
            val_arr = np.array(data_exp1[p]["leakage"])
            idx = np.argsort(qber_arr)
            plt.plot(qber_arr[idx], val_arr[idx], styles[p]['marker'], color=styles[p]['color'], label=styles[p]['label'])
    plt.xlabel('Measured QBER (%)')
    plt.ylabel('Key Leakage (% of Sifted Key)')
    plt.title('Information Leakage Comparison')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.outdir, "1_leakage_vs_qber.png"))
    #plt.savefig(os.path.join(args.outdir,"plots_eps/1_leakage_vs_qber.eps"), format='eps', dpi=600, bbox_inches='tight')
    plt.close()

    # Plot 2: Efficiency
    plt.figure(figsize=(10, 6))
    for p in args.protocols:
        if data_exp1[p]["qber"]:
            qber_arr = np.array(data_exp1[p]["qber"])
            val_arr = np.array(data_exp1[p]["efficiency"])
            idx = np.argsort(qber_arr)
            plt.plot(qber_arr[idx], val_arr[idx], styles[p]['marker'], color=styles[p]['color'], label=styles[p]['label'])
    plt.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Shannon Limit')
    plt.xlabel('Measured QBER (%)')
    plt.ylabel(r'Reconciliation Inefficiency $\eta$')
    plt.title('Protocol Inefficiency')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.outdir, "2_efficiency_vs_qber.png"))
    #plt.savefig(os.path.join(args.outdir,"plots_eps/2_efficiency_vs_qber.eps"), format='eps', dpi=600, bbox_inches='tight')
    plt.close()

    # Plot 3: Latency
    plt.figure(figsize=(10, 6))
    for p in args.protocols:
        if data_exp1[p]["qber"]:
            qber_arr = np.array(data_exp1[p]["qber"])
            val_arr = np.array(data_exp1[p]["uses"])
            idx = np.argsort(qber_arr)
            plt.plot(qber_arr[idx], val_arr[idx], styles[p]['marker'], color=styles[p]['color'], label=styles[p]['label'])
    plt.yscale('log')
    plt.xlabel('Measured QBER (%)')
    plt.ylabel('Channel Uses (Round Trips) - Log Scale')
    plt.title('Communication Latency Comparison')
    plt.legend(); plt.grid(True, which="both", alpha=0.3)
    plt.savefig(os.path.join(args.outdir, "3_latency_vs_qber.png"))
    #plt.savefig(os.path.join(args.outdir,"plots_eps/3_latency_vs_qber.eps"), format='eps', dpi=600, bbox_inches='tight')
    plt.close()

    # Plot 4: Scalability
    plt.figure(figsize=(10, 6))
    for p in args.protocols:
        plt.plot(block_sizes, data_exp2[p], styles[p]['marker'], color=styles[p]['color'], label=styles[p]['label'])
    plt.xlabel('Block Size (Bits)')
    plt.ylabel('Execution Time (Seconds)')
    plt.title(f'Computational Scalability (Avg over {args.runs} runs, Err ~{FIXED_ERR})')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.outdir, "4_execution_time_vs_blocksize.png"))
    #plt.savefig(os.path.join(args.outdir,"plots_eps/4_execution_time_vs_blocksize.eps"), format='eps', dpi=600, bbox_inches='tight')
    plt.close()

    # Plot 5: Key Rate
    plt.figure(figsize=(10, 6))
    for p in args.protocols:
        if data_exp1[p]["qber"]:
            qber_arr = np.array(data_exp1[p]["qber"])
            eff_arr = np.array(data_exp1[p]["efficiency"])
            key_rate = np.array([1 - (1 + eff) * binary_entropy(q/100) for eff, q in zip(eff_arr, qber_arr)])
            idx = np.argsort(qber_arr)
            plt.plot(qber_arr[idx], key_rate[idx], styles[p]['marker'], color=styles[p]['color'], label=styles[p]['label'])
    plt.xlabel('Measured QBER (%)')
    plt.ylabel('Secure Key Rate')
    plt.title('Secure Key Rate Fraction vs QBER')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(args.outdir, "5_secure_key_rate_vs_qber.png"))
    #plt.savefig(os.path.join(args.outdir,"plots_eps/5_secure_key_rate_vs_qber.eps"), format='eps', dpi=600, bbox_inches='tight')
    plt.close()

    print(f"All plots saved to ./{args.outdir}")
    print("Done!")

if __name__ == "__main__":
    asyncio.run(main())
