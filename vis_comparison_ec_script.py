import asyncio
import random
import sys
import os
import math
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
import hashlib
from typing import List

# Setup proper imports
current_dir = os.getcwd()
target_dir = os.path.join(current_dir, 'ServerClientBB84')
if target_dir not in sys.path:
    sys.path.append(target_dir)

try:
    from ServerClientBB84.bb84_server_client import AliceServer, BobClient, QuantumChannel, Detector, Eve, APIClient
    from ServerClientBB84.privacy_amplification import PrivacyAmplification
except ImportError:
    sys.path.append(os.getcwd())
    from ServerClientBB84.bb84_server_client import AliceServer, BobClient, QuantumChannel, Detector, Eve, APIClient

# ---- Global plot styling for research paper ----
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 14,
    'figure.titlesize': 16,
    'figure.figsize': (10, 6),
    'lines.linewidth': 2,
    'lines.markersize': 8,
})


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
    
    alice = AliceServer("Alice", channel, num_qubits=num_qubits,mu=0.1, verbose=verbose, seed=seed_base+1)
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
        # if "exec_time" not in results:
        #     results["exec_time"] = time.time() - start_time
        
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
        "sifted_length": np.mean([r['sifted_length'] for r in valid_results]),
        # --- Per-run quantities (so caller averages ratios, not means) ---
        "eff_per_run": [
            r['revealed'] / (r['final_length'] * binary_entropy(r['qber']))
            for r in valid_results
            if r['qber'] > 0 and r['final_length'] > 0
        ],
        "leakage_per_run": [
            (r['revealed'] / r['final_length']) * 100.0
            for r in valid_results
            if r['final_length'] > 0
        ],
        "skf_per_run": [
        r.get('pa_length', 0) / r['sifted_length']
        for r in valid_results
        if r['sifted_length'] > 0
        ],
    }
    return avg_res



def plot_pa_protocol_comparison(
    key_lengths=(256, 512, 1024, 2048, 4096, 8192),
    qber=0.035,
    security_parameter=10,
    q1=1.0,
    ec_leakage=0,
    repeats=10,
    outpath="pa_toeplitz_vs_shake256",
    save_eps=True,
    save_png=True,
    show=True,
    font_scale=1.6,
):
    """
    Single-panel benchmark: Toeplitz vs SHAKE256 privacy amplification.

    Left y-axis  : execution time (s), averaged over `repeats` runs.
    Right y-axis : public randomness / seed bits required.

    - Toeplitz needs N + R - 1 random bits (information-theoretic universal hash).
    - SHAKE256 uses a fixed 256-bit salt (computational security).
    """
    # ---- Styling for research paper ----
    # base = 14 * font_scale
    # plt.rcParams.update({
    #     "font.size": base,
    #     "axes.labelsize": base + 2,
    #     "axes.titlesize": base + 3,
    #     "xtick.labelsize": base,
    #     "ytick.labelsize": base,
    #     "legend.fontsize": base - 3.5,
    #     "figure.titlesize": base + 4,
    # })

    toe_times, shake_times = [], []
    toe_seed_bits, shake_seed_bits = [], []
    final_key_lengths = []

    for N in key_lengths:
        print(f"  [PA] N={N} ...", flush=True)
        key = [random.randint(0, 1) for _ in range(N)]
        pa = PrivacyAmplification(
            key=key,
            qber=qber,
            security_parameter=security_parameter,
            q1=q1,
            ec_leakage=ec_leakage,
        )
        R = pa.R
        final_key_lengths.append(R)

        # --- Toeplitz ---
        runs = []
        for r in range(repeats):
            t0 = time.perf_counter()
            pa.apply_hash(seed=12345 + r, algorithm="toeplitz")
            runs.append(time.perf_counter() - t0)
        toe_times.append(sum(runs) / len(runs))
        print(f"    Toeplitz: {toe_times[-1]*1e3:.2f} ms", flush=True)

        # --- SHAKE256 ---
        runs = []
        for r in range(repeats):
            t0 = time.perf_counter()
            pa.apply_hash(seed=12345 + r, algorithm="shake256")
            runs.append(time.perf_counter() - t0)
        shake_times.append(sum(runs) / len(runs))
        print(f"    SHAKE256: {shake_times[-1]*1e3:.2f} ms", flush=True)

        toe_seed_bits.append(N + R - 1 if R > 0 else 0)
        shake_seed_bits.append(256)

    # ---- Single figure, dual y-axes ----
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    color_toe = "#1f77b4"
    color_shake = "#d62728"

    # Runtime (left axis)
    l1, = ax1.plot(key_lengths, toe_times, "o-", color=color_toe,
                   markersize=9, linewidth=2.2, label="Toeplitz(time)")
    l2, = ax1.plot(key_lengths, shake_times, "s-", color=color_shake,
                   markersize=9, linewidth=2.2, label="SHAKE256(time)")

    # Seed bits (right axis)
    l3, = ax2.plot(key_lengths, toe_seed_bits, "^--", color=color_toe,
                   alpha=0.55, markersize=9, linewidth=2.0,
                   label="Toeplitz(seed bits)")
    l4, = ax2.plot(key_lengths, shake_seed_bits, "v--", color=color_shake,
                   alpha=0.55, markersize=9, linewidth=2.0,
                   label="SHAKE256(seed bits)")

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax2.set_yscale("log")

    # FIX 1: Use $...$ instead of \(...\)
    ax1.set_xlabel("Reconciled key length (bits)")
    ax1.set_ylabel("Execution time (s)")
    ax2.set_ylabel("Public seed bits")

    # FIX 2: Remove the old grid line. 
    # Turn on only major gridlines, and explicitly turn off minor ones.
    ax1.grid(True, which="major", alpha=0.5, linestyle="-")
    ax1.grid(False, which="minor")

    # Merge legends
    lines = [l1, l2, l3, l4]
    ax1.legend(lines, [l.get_label() for l in lines],
               loc="upper left", framealpha=0.95)

    # tight_layout should always be called last, after all labels/grids are set
    fig.tight_layout()

    # ---- Save ----
    saved = []
    if save_eps:
        eps_path = f"{outpath}.eps"
        fig.savefig(eps_path, format="eps", dpi=600, bbox_inches="tight")
        saved.append(eps_path)
    if save_png:
        png_path = f"{outpath}.png"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        saved.append(png_path)
    print(f"[plot] Saved: {saved}")

    # if show:
    #     plt.show()
    

    return {
        "key_lengths": list(key_lengths),
        "final_key_lengths": final_key_lengths,
        "toeplitz_times": toe_times,
        "shake256_times": shake_times,
        "toeplitz_seed_bits": toe_seed_bits,
        "shake256_seed_bits": shake_seed_bits,
    }

#old0 - num_iter=30, max_k=8000, target_fer=1e-2, sweep_factor=1.08, max_sweep_iter=10, hash_bits=32; 
# old1-6000,1e-3,10iter,50llrparity ; 
# new1-8000,1e-2,10iter,50llrparity
async def main():
    parser = argparse.ArgumentParser(description="BB84 Reconciliation Comparison Script")
    parser.add_argument("--runs", type=int, default=100, help="Number of Monte Carlo iterations to average per data point")
    parser.add_argument("--qubits", type=int, default=100000, help="Initial sequence size for QBER sweep")
    parser.add_argument("--outdir", type=str, default="figures/serverPlots_Finiteasym/new3/", help="Directory to save the resulting plots")
    parser.add_argument("--protocols", nargs='+', default=["cascade", "nr_ldpc_std",  "winnow", "polar"],  #"ldpc_rateadaptive",
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
        "nr_ldpc_std": {'color': 'purple', 'marker': 's-', 'label': 'LDPC'},
        "winnow": {'color': 'green', 'marker': '^-', 'label': 'Winnow'},
        "polar": {'color': 'orange', 'marker': 'd-', 'label': 'Polar Codes'}
    }

    # Data stores
    data_exp1 = {p: {"qber": [], "leakage": [], "efficiency": [], "uses": [], "skf": []} for p in args.protocols}
    
    # ---------------- EXPERIMENT 1 ----------------
    print("\n--- Starting Experiment 1: Impact of QBER ---")
    error_rates = np.linspace(0.001, 0.12, 25) 
    
    for err in error_rates:
        print(f"\n[Exp 1] Testing Optical Error: {err:.4f}")
        for protocol in args.protocols:
            avg_res = await run_multiple_simulations(protocol, args.qubits, err, args.runs)
            
            if avg_res and avg_res.get('sifted_length', 0) > 0 and avg_res.get('qber', 0) > 0:
                # eff = calculate_efficiency(avg_res['revealed'], avg_res['final_length'], avg_res['qber'])
                # if eff:
                #     data_exp1[protocol]["qber"].append(avg_res['qber'] * 100)
                #     data_exp1[protocol]["leakage"].append((avg_res['revealed'] / avg_res['final_length']) * 100)
                #     data_exp1[protocol]["efficiency"].append(eff)
                #     data_exp1[protocol]["uses"].append(avg_res['channel_uses'])
                effs  = avg_res.get("eff_per_run", [])
                leaks = avg_res.get("leakage_per_run", [])
                skfs  = avg_res.get("skf_per_run", []) 
                if effs:
                    data_exp1[protocol]["qber"].append(avg_res['qber'] * 100)
                    data_exp1[protocol]["leakage"].append(float(np.mean(leaks)))
                    data_exp1[protocol]["efficiency"].append(float(np.mean(effs)))
                    data_exp1[protocol]["uses"].append(avg_res['channel_uses'])
                    data_exp1[protocol]["skf"].append(float(np.mean(skfs)))
            
            print(f"  > {protocol}: avg QBER={avg_res['qber']*100:.2f}%" if avg_res else f"  > {protocol}: Failed")

    # # ---------------- EXPERIMENT 2 ----------------
    print("\n--- Starting Experiment 2: Computation Time vs Block Size ---")
    block_sizes = [1000, 2500, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]
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
    plt.ylabel('Key Leakage (% of Reconciled Key)')
    #plt.title('Information Leakage Comparison')
    plt.legend(); plt.grid(True, alpha=0.3)
    #plt.savefig(os.path.join(args.outdir, "1_leakage_vs_qber.png"))
    plt.savefig(os.path.join(args.outdir,"plots_eps/1_leakage_vs_qber.eps"), format='eps', dpi=600, bbox_inches='tight')
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
    plt.ylabel(r'Reconciliation Efficiency $f_{EC}$')
    #plt.title('Protocol Inefficiency')
    plt.legend(fontsize=9)
    plt.ylim(0.8, 3)
    plt.grid(True, alpha=0.3)
    #plt.savefig(os.path.join(args.outdir, "2_efficiency_vs_qber.png"))
    plt.savefig(os.path.join(args.outdir,"plots_eps/2_efficiency_vs_qber.eps"), format='eps', dpi=600, bbox_inches='tight')
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
    #plt.title('Communication Latency Comparison')
    # plt.legend(); 
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(
        fontsize=9,           # text size inside the box
    )
    #plt.savefig(os.path.join(args.outdir, "3_latency_vs_qber.png"))
    plt.savefig(os.path.join(args.outdir,"plots_eps/3_latency_vs_qber.eps"), format='eps', dpi=600, bbox_inches='tight')
    plt.close()

    # Plot 4: Scalability
    plt.figure(figsize=(10, 6))
    for p in args.protocols:
        plt.plot(block_sizes, data_exp2[p], styles[p]['marker'], color=styles[p]['color'], label=styles[p]['label'])
    plt.xlabel('Block Size (Bits)')
    plt.ylabel('Execution Time (Seconds)')
    #plt.title(f'Computational Scalability (Avg over {args.runs} runs, Err ~{FIXED_ERR})')
    plt.legend(); plt.grid(True, alpha=0.3)
    #plt.savefig(os.path.join(args.outdir, "4_execution_time_vs_blocksize.png"))
    plt.savefig(os.path.join(args.outdir,"plots_eps/4_execution_time_vs_blocksize.eps"), format='eps', dpi=600, bbox_inches='tight')
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
    plt.ylabel('Secret Key Fraction')
    #plt.title('Secure Key Rate Fraction vs QBER')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    #plt.savefig(os.path.join(args.outdir, "5_secure_key_rate_vs_qber.png"))
    plt.savefig(os.path.join(args.outdir,"plots_eps/5_secure_key_rate_vs_qber.eps"), format='eps', dpi=600, bbox_inches='tight')
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
    plt.ylabel('Estimated Secret Key Fraction')
    #plt.title('Secure Key Rate Fraction vs QBER')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    #plt.savefig(os.path.join(args.outdir, "5_secure_key_rate_vs_qber.png"))
    plt.savefig(os.path.join(args.outdir,"plots_eps/5_5_estimate_secure_key_fraction_vs_qber.eps"), format='eps', dpi=600, bbox_inches='tight')
    plt.close()

    # Plot 6: Realized Secret Key Fraction
    plt.figure(figsize=(10, 6))
    for p in args.protocols:
        if data_exp1[p]["qber"]:
            qber_arr = np.array(data_exp1[p]["qber"])
            val_arr  = np.array(data_exp1[p]["skf"])
            idx = np.argsort(qber_arr)
            plt.plot(qber_arr[idx], val_arr[idx],
                    styles[p]['marker'], color=styles[p]['color'],
                    label=styles[p]['label'])
    plt.xlabel('Measured QBER (%)')
    plt.ylabel(r'Realized Secret Key Fraction')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(args.outdir, "plots_eps/6_realized_skf_vs_qber.eps"),
                format='eps', dpi=600, bbox_inches='tight')
    plt.close()


    print(f"All plots saved to ./{args.outdir}")
    print("Done!")

    estimated_qber = FIXED_ERR

    # #Compare Toeplitz vs SHAKE256 privacy amplification
    # plot_pa_protocol_comparison(
    #     key_lengths=(100,500, 1000,5000, 10000, 50000, 100000),
    #     qber=estimated_qber,
    #     security_parameter=10,
    #     repeats=10,                       # averaged over 10 runs
    #     outpath="figures/serverPlots_Finiteasym/plots_eps/pa_toeplitz_vs_shake256",
    #     save_eps=True,                    # writes .eps (600 dpi)
    #     save_png=False,                    # also writes .png
    #     show=True,
    #     font_scale=1.6,                   # bump this to enlarge fonts further
    # )


if __name__ == "__main__":
    asyncio.run(main())
