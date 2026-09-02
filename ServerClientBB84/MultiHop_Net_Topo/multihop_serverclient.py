import sys
import os
import asyncio
import time
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import ServerClientBB84.bb84_server_client as sc
import bb84_finite as finite


def _figure_to_base64(fig):
    import io
    import base64

    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight', dpi=120)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')


class ProtocolWrapperServerClient:
    """Wrapper that runs a lightweight server-client BB84 simulation
    and returns an effective secret-bit rate (bits/sec) for a single
    link of specified distance.
    """
    def __init__(self, protocol='polar', pa_protocol='toeplitz', num_qubits=5000,
                 freq=1e7, mu=1.0, att_db_km=0.2, detector_eff=0.8, dark_count=0.01,
                 protocol_params=None, verbose=False, seeds=None):
        self.protocol = protocol
        self.pa_protocol = pa_protocol
        self.num_qubits = int(num_qubits)
        self.freq = freq
        self.mu = mu
        self.att_db_km = att_db_km
        self.det_eff = detector_eff
        self.dark_count = dark_count
        self.protocol_params = protocol_params if protocol_params else {}
        self.verbose = verbose
        # Optional seeds dict: {'alice':..,'bob':..,'channel':..,'detector':..}
        self.seeds = seeds or {}
        # Number of repeated trials to average per distance (reduces statistical spikes)
        self.num_trials = 1

    def calculate_rate(self, distance_km):
        """Run one full server-client simulation for a single link.

        Returns: (rate_bps, metrics_dict)
        """
        # Allow averaging over repeated independent trials to reduce noise
        if self.num_trials <= 1:
            try:
                return asyncio.run(self._run_single_link(distance_km, None))
            except Exception as e:
                # Return zero rate and diagnostic metrics instead of raising
                err_metrics = {'error': str(e)}
                if self.verbose:
                    print(f"[ProtocolWrapper] Single-run failure at {distance_km} km: {e}")
                return 0.0, err_metrics

        # Run multiple trials with varied seeds and average the resulting rates
        rates = []
        metrics_list = []

        # Determine base seeds; if none supplied use defaults
        base_seeds = self.seeds if self.seeds else {'alice': 1001, 'bob': 1002, 'channel': 1003, 'detector': 1004}

        for t in range(self.num_trials):
            # Derive per-trial seeds by offsetting the base seeds
            trial_seeds = {k: int(v) + t for k, v in base_seeds.items()}
            try:
                rate, metrics = asyncio.run(self._run_single_link(distance_km, trial_seeds))
            except Exception as e:
                rate = 0.0
                metrics = {'error': str(e), 'trial': t, 'distance_km': distance_km}
                if self.verbose:
                    print(f"[ProtocolWrapper] Trial {t} failed at {distance_km} km: {e}")
            rates.append(rate)
            metrics_list.append(metrics)

        avg_rate = float(np.mean(rates))
        return avg_rate, metrics_list

    async def _run_single_link(self, distance_km, seeds_override=None):
        # Prepare seeds
        seeds = seeds_override if seeds_override is not None else self.seeds
        seed_alice = seeds.get('alice', 1001)
        seed_bob = seeds.get('bob', 1002)
        seed_channel = seeds.get('channel', 1003)
        seed_detector = seeds.get('detector', 1004)

        # Seed NumPy RNG for reproducible Poisson draws (photon counts)
        # Use Alice seed so behavior is repeatable across runs unless user changes seeds
        np.random.seed(int(seed_alice))

        # Create channel and actors
        channel = sc.QuantumChannel("Fiber", distance_km, self.att_db_km, 0.01, next_actor=None, seed=seed_channel)
        alice = sc.AliceServer("Alice", channel, num_qubits=self.num_qubits, mu=self.mu, verbose=self.verbose, seed=seed_alice)
        api = sc.APIClient(alice)
        bob = sc.BobClient("Bob", api, protocol=self.protocol, pa_protocol=self.pa_protocol, seed=seed_bob, verbose=self.verbose, protocol_params=self.protocol_params)
        detector = sc.Detector("Detector", self.det_eff, self.dark_count, parent_bob=bob, seed=seed_detector)

        # Wire channel to detector
        channel.next_actor = detector

        # Start actor loops
        actors = [alice, channel, detector, bob]
        tasks = [asyncio.create_task(a.start()) for a in actors]

        # Transmission phase
        t_tx_start = time.time()
        await alice.run_quantum_transmission()

        # Wait for qubits to propagate and be processed
        # Use a conservative wait: transmission duration + small margin
        tx_duration = max(0.1, self.num_qubits / max(self.freq, 1.0))
        await asyncio.sleep(tx_duration + 0.1)

        # Classical post-processing
        # Compute q1 like the server-client script does
        T = channel.transmittance
        eta = detector.eta
        mu = alice.mu
        p_sig = 1.0 - np.exp(-mu * T * eta)
        p_dark = 2 * detector.p_dc
        p_click = p_sig + p_dark - (p_sig * p_dark)
        p_multi = 1.0 - (1.0 + mu) * np.exp(-mu)
        q1 = max(0.0, 1.0 - (p_multi / p_click)) if p_click > 0 else 0.0

        metrics = await bob.run_classical_post_processing(alice.num_qubits, q1=q1)

        # Stop actors
        for a in actors:
            await a.send(a, ("STOP",))
        await asyncio.gather(*tasks)

        # Calculate effective bits and rate
        # Prefer PA output length, otherwise corrected key length
        final_bits = 0
        if metrics.get('pa_length', 0) > 0 and 'final_secret_key' in metrics:
            final_bits = len(metrics['final_secret_key'])
        else:
            final_bits = metrics.get('final_length', 0)

        total_time = tx_duration + metrics.get('exec_time', 0) + metrics.get('pa_time', 0)
        if total_time <= 0:
            rate = 0.0
        else:
            rate = final_bits / total_time

        return rate, metrics


class MultiHopChain:
    def __init__(self, sifting_exchanges=3, overhead_factor=3, packet_size=10000):
        self.S = sifting_exchanges
        self.n = overhead_factor
        self.c = 3e5
        self.L_packet = packet_size

    def calculate_chain_rate(self, link_rates, link_distances, link_capacities):
        if not link_rates or any(k <= 1e-9 for k in link_rates):
            return 0.0
        bottleneck_rate = min(link_rates)
        t_generation_bottleneck = (2 * self.L_packet) / bottleneck_rate
        latency_total = 0
        for i in range(len(link_rates)):
            D = link_distances[i]
            C = link_capacities[i]
            t_prop = (self.S * D) / self.c
            t_trans = (self.n * self.L_packet) / C
            latency_total += t_prop + t_trans
        total_tau = t_generation_bottleneck + latency_total
        return self.L_packet / total_tau


def simulate_optimal_hops(
    protocols=None,
    pa_protocols=None,
    total_distance=500.0,
    relay_counts=None,
    num_qubits=5000,
    link_capacity=1e9,
    sim_config=None,
    chain_params=None
):
    if protocols is None:
        protocols = ['polar']
    if pa_protocols is None:
        pa_protocols = ['toeplitz']
    if relay_counts is None:
        relay_counts = np.linspace(1, 15, 20, dtype=int).tolist()
    if sim_config is None: sim_config = {}
    if chain_params is None: chain_params = {}

    proto_common = dict(sim_config)
    results = {}

    for proto in protocols:
        for pa in pa_protocols:
            cfg = dict(proto_common)
            cfg['protocol'] = proto
            cfg['pa_protocol'] = pa

            print(f"   -> Simulating Optimal Hops for ({proto}/{pa})...")

            wrapper = ProtocolWrapperServerClient(
                protocol=proto, pa_protocol=pa, num_qubits=num_qubits,
                freq=cfg.get('freq', 1e7), mu=cfg.get('mu', 1.0), att_db_km=cfg.get('att_db_km', 0.2), 
                detector_eff=cfg.get('det_eff', 0.8), dark_count=cfg.get('dark_count', 0.01), 
                protocol_params=cfg.get('protocol_params', {}), verbose=False, seeds=cfg.get('seeds', {})
            )
            # Configure averaging and seeds
            wrapper.num_trials = int(cfg.get('num_trials', 10))
            if not wrapper.seeds:
                base_seed = int(cfg.get('base_seed', int(time.time()) % 100000))
                wrapper.seeds = {'alice': base_seed + 1, 'bob': base_seed + 2, 'channel': base_seed + 3, 'detector': base_seed + 4}

            chain_logic = MultiHopChain(**chain_params)
            final_rates = []

            for num_relays in relay_counts:
                num_links = num_relays + 1
                dist_per_link = total_distance / num_links

                rate_bps, metrics = wrapper.calculate_rate(dist_per_link)

                if rate_bps <= 1e-9:
                    final_rates.append(0.0)
                    continue

                rates = [rate_bps] * num_links
                dists = [dist_per_link] * num_links
                caps = [link_capacity] * num_links

                eff_rate = chain_logic.calculate_chain_rate(rates, dists, caps)
                final_rates.append(eff_rate)

            results[f'{proto}/{pa}'] = final_rates

    return relay_counts, results


def simulate_range_extension(
    max_distance=500,
    num_points=20,
    comparison_relays=3,
    num_qubits=5000,
    link_capacity=1e9,
    sim_config=None,
    chain_params=None
):
    if sim_config is None: sim_config = {}
    if chain_params is None: chain_params = {}

    proto = sim_config.get('protocol', 'polar')
    pa = sim_config.get('pa_protocol', 'toeplitz')
    freq = sim_config.get('freq', 1e7)
    mu = sim_config.get('mu', 1.0)
    att = sim_config.get('att_db_km', 0.2)
    det_eff = sim_config.get('det_eff', 0.8)
    dark = sim_config.get('dark_count', 0.01)
    protocol_params = sim_config.get('protocol_params', {})
    seeds = sim_config.get('seeds', {})

    wrapper = ProtocolWrapperServerClient(
        protocol=proto, pa_protocol=pa, num_qubits=num_qubits,
        freq=freq, mu=mu, att_db_km=att, detector_eff=det_eff,
        dark_count=dark, protocol_params=protocol_params, verbose=False, seeds=seeds
    )
    # Configure averaging and seeds
    wrapper.num_trials = int(sim_config.get('num_trials', 10))
    if not wrapper.seeds:
        base_seed = int(sim_config.get('base_seed', int(time.time()) % 100000))
        wrapper.seeds = {'alice': base_seed + 1, 'bob': base_seed + 2, 'channel': base_seed + 3, 'detector': base_seed + 4}

    chain_logic = MultiHopChain(**chain_params)

    scan_distances = np.linspace(10, max_distance, num_points)
    rates_direct = []
    rates_relayed = []

    num_links_relayed = comparison_relays + 1

    print(f"   -> Simulating Range Extension ({proto}/{pa})...")

    for D in scan_distances:
        rate_direct, _ = wrapper.calculate_rate(D)
        r_direct = chain_logic.calculate_chain_rate([rate_direct], [D], [link_capacity])
        rates_direct.append(r_direct)

        d_seg = D / num_links_relayed
        rate_seg, _ = wrapper.calculate_rate(d_seg)
        r_chain = chain_logic.calculate_chain_rate([rate_seg] * num_links_relayed, [d_seg] * num_links_relayed, [link_capacity] * num_links_relayed)
        rates_relayed.append(r_chain)

    return scan_distances, rates_direct, rates_relayed


def simulate_range_extension_protocols(
    protocols=None,
    pa_protocols=None,
    max_distance=500,
    num_points=20,
    comparison_relays=3,
    num_qubits=5000,
    link_capacity=1e9,
    sim_config=None,
    chain_params=None
):
    """Run range-extension sweeps for multiple EC and PA protocol combinations and return results.

    Returns: (distances, results_dict) where results_dict['ec/pa'] = (direct_rates, relayed_rates)
    """
    if protocols is None:
        protocols = ['polar', 'cascade', 'nr_ldpc_standard', 'ldpc_rateadaptive', 'winnow']
    if pa_protocols is None:
        pa_protocols = ['toeplitz']
    if sim_config is None:
        sim_config = {}
    if chain_params is None:
        chain_params = {}

    proto_common = dict(sim_config)
    scan_distances = np.linspace(10, max_distance, num_points)
    results = {}

    # Compute a single BB84 finite direct-link curve for comparison
    try:
        f_src = finite.Source(freq=proto_common.get('freq', 1e7), mean_photon_num=proto_common.get('mu', 0.1), q=1.0, alignment_error=0.0)
        f_det = finite.Detector(efficiency=proto_common.get('det_eff', 0.8), dark_count_rate=1e4, time_window=1e-9)
        f_chan = finite.Channel(att_db_km=proto_common.get('att_db_km', 0.2))
        f_proto = finite.Protocol(f_src, f_chan, f_det)
        finite_direct_rates = f_proto.skr_vs_distance(scan_distances, fixed_N=num_qubits)
        # Ensure numeric numpy array and provide quick diagnostics so user can see if values are tiny/zero
        finite_direct_rates = np.asarray(finite_direct_rates, dtype=float)
        try:
            mn = finite_direct_rates.min()
            mx = finite_direct_rates.max()
            any_nonzero = np.any(finite_direct_rates > 0)
            print(f"[multihop] bb84_finite direct curve: min={mn:.3e}, max={mx:.3e}, any_nonzero={any_nonzero}")
            # If all-zero, it's likely the chosen mean-photon number (mu) is too large
            # so the single-photon fraction q1 becomes zero; try a smaller mu for debug
            if not any_nonzero:
                debug_mu = 0.1
                print(f"[multihop] finite-direct all zero — retrying with mu={debug_mu} for diagnosis")
                f_src2 = finite.Source(freq=proto_common.get('freq', 1e7), mean_photon_num=debug_mu, q=1.0, alignment_error=0.0)
                f_proto2 = finite.Protocol(f_src2, f_chan, f_det)
                finite_direct_rates2 = np.asarray(f_proto2.skr_vs_distance(scan_distances, fixed_N=num_qubits), dtype=float)
                mn2 = finite_direct_rates2.min()
                mx2 = finite_direct_rates2.max()
                any_nonzero2 = np.any(finite_direct_rates2 > 0)
                print(f"[multihop] retry(mu={debug_mu}) : min={mn2:.3e}, max={mx2:.3e}, any_nonzero={any_nonzero2}")
                # If retry produced values, use them and warn the user about mu choice
                if any_nonzero2:
                    print(f"[multihop] Using retry mu={debug_mu} finite-direct curve (original mu likely too large)")
                    finite_direct_rates = finite_direct_rates2
        except Exception:
            pass
    except Exception as e:
        finite_direct_rates = np.zeros_like(scan_distances, dtype=float)
        if 'verbose' in proto_common and proto_common.get('verbose'):
            print(f"[multihop] Failed to compute bb84_finite direct curve: {e}")

    for proto in protocols:
        for pa in pa_protocols:
            cfg = dict(proto_common)
            cfg['protocol'] = proto
            cfg['pa_protocol'] = pa

            wrapper = ProtocolWrapperServerClient(
                protocol=proto, pa_protocol=pa, num_qubits=num_qubits,
                freq=cfg.get('freq',1e7), mu=cfg.get('mu',1.0), att_db_km=cfg.get('att_db_km',0.2),
                detector_eff=cfg.get('det_eff',0.8), dark_count=cfg.get('dark_count',0.01),
                protocol_params=cfg.get('protocol_params', {}), verbose=False, seeds=cfg.get('seeds', {})
            )
            wrapper.num_trials = int(cfg.get('num_trials', 10))

            chain = MultiHopChain(**chain_params)
            relayed_rates = []

            num_links_relayed = comparison_relays + 1

            for D in scan_distances:
                # relayed
                d_seg = D / num_links_relayed
                rate_seg, _ = wrapper.calculate_rate(d_seg)
                r_chain = chain.calculate_chain_rate([rate_seg] * num_links_relayed, [d_seg] * num_links_relayed, [link_capacity] * num_links_relayed)
                relayed_rates.append(r_chain)

            # Store the same finite direct_rates for each combination to keep return shape consistent
            results[f'{proto}/{pa}'] = (finite_direct_rates.tolist(), relayed_rates)

    return scan_distances, results


def plot_range_extension_protocols(distances, results, relays, config=None):
    """Plot direct vs relayed curves for multiple protocol combinations side-by-side."""
    plt.figure(figsize=(12, 7))
    linestyles = {'direct': '--', 'relayed': '-'}
    colors = plt.cm.get_cmap('tab10')
    protocols = list(results.keys())

    # Do not plot the bb84_finite direct curve here (removed per request)
    all_positive_vals = []

    # Plot relayed curves per protocol combo
    for i, combo in enumerate(protocols):
        _, relayed_rates = results[combo]
        color = colors(i % 10)
        relayed_arr = np.asarray(relayed_rates, dtype=float)
        plt.plot(distances, relayed_arr, linestyle=linestyles['relayed'], color=color, linewidth=2, label=f'{combo} (relayed)')
        all_positive_vals.append(relayed_arr[relayed_arr > 0])

    plt.xlabel('Total Distance Alice-Bob (km)')
    plt.ylabel('Effective Key Rate (bps)')
    plt.title(f'Range Extension: Direct vs Relayed — Protocol Comparison (Relays={relays})')
    plt.yscale('log')
    plt.legend(fontsize='small', ncol=2)
    plt.grid(True, which='both', ls='--', alpha=0.6)
    # Choose a sensible lower y-limit for log scale so small but nonzero rates remain visible
    try:
        positives = np.hstack([a for a in all_positive_vals if a.size > 0]) if all_positive_vals else np.array([])
        if positives.size > 0:
            y_min_pos = positives.min()
            plt.ylim(bottom=max(y_min_pos * 0.1, 1e-12))
    except Exception:
        pass
    plt.show()


def create_optimal_hops_figure(x_data, results, total_dist, title=None):
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.get_cmap('tab10')

    if title is None:
        title = f'Optimal Hop Analysis — Protocol Comparison\nTotal Dist={total_dist}km'

    all_positive_vals = []

    for i, (combo_name, y_data) in enumerate(results.items()):
        color = colors(i % 10)
        y_arr = np.asarray(y_data, dtype=float)
        ax.plot(x_data, y_arr, 'o-', color=color, linewidth=2, label=combo_name)
        all_positive_vals.append(y_arr[y_arr > 0])

    ax.set_xlabel('Number of Intermediate Relays')
    ax.set_ylabel('Effective End-to-End Rate (bps)')
    ax.set_title(title)
    ax.grid(True, which='both', ls='--', alpha=0.7)
    ax.set_yscale('log')
    ax.legend(fontsize='small', ncol=2)
    
    try:
        positives = np.hstack([a for a in all_positive_vals if a.size > 0]) if all_positive_vals else np.array([])
        if positives.size > 0:
            y_min_pos = positives.min()
            ax.set_ylim(bottom=max(y_min_pos * 0.1, 1e-12))
    except Exception:
        pass
    
    return fig


def create_range_extension_protocols_figure(distances, results, relays, config=None):
    fig, ax = plt.subplots(figsize=(12, 7))
    linestyles = {'relayed': '-'}
    colors = plt.cm.get_cmap('tab10')
    protocols = list(results.keys())

    all_positive_vals = []

    for i, combo in enumerate(protocols):
        _, relayed_rates = results[combo]
        color = colors(i % 10)
        relayed_arr = np.asarray(relayed_rates, dtype=float)
        ax.plot(distances, relayed_arr, linestyle=linestyles['relayed'], color=color, linewidth=2, label=f'{combo} (relayed)')
        all_positive_vals.append(relayed_arr[relayed_arr > 0])

    ax.set_xlabel('Total Distance Alice-Bob (km)')
    ax.set_ylabel('Effective Key Rate (bps)')
    ax.set_title(f'Range Extension: Direct vs Relayed — Protocol Comparison (Relays={relays})')
    ax.set_yscale('log')
    ax.legend(fontsize='small', ncol=2)
    ax.grid(True, which='both', ls='--', alpha=0.6)
    try:
        positives = np.hstack([a for a in all_positive_vals if a.size > 0]) if all_positive_vals else np.array([])
        if positives.size > 0:
            y_min_pos = positives.min()
            ax.set_ylim(bottom=max(y_min_pos * 0.1, 1e-12))
    except Exception:
        pass
    return fig


def simulate_range_extension_protocol_pairs(
    chain_data,
    protocols=None,
    pa_protocols=None,
    max_distance=500,
    num_points=20,
    num_qubits=5000,
    link_capacity=1e9,
    sim_config=None,
    chain_params=None,
):
    if protocols is None:
        protocols = ['polar', 'cascade', 'nr_ldpc_standard', 'ldpc_rateadaptive', 'winnow']
    if pa_protocols is None:
        pa_protocols = ['toeplitz']
    if sim_config is None:
        sim_config = {}
    if chain_params is None:
        chain_params = {}

    proto_common = dict(sim_config)
    scan_distances = np.linspace(10, max_distance, num_points)
    results = {}

    usable_chain = chain_data or [{'distance': max_distance, 'capacity': link_capacity}]
    total_chain_distance = sum(float(seg.get('distance', 0.0)) for seg in usable_chain)
    if total_chain_distance <= 0:
        total_chain_distance = max_distance if max_distance > 0 else 1.0

    segment_ratios = [float(seg.get('distance', 0.0)) / total_chain_distance for seg in usable_chain]
    if not segment_ratios:
        segment_ratios = [1.0]

    for proto in protocols:
        for pa in pa_protocols:
            cfg = dict(proto_common)
            cfg['protocol'] = proto
            cfg['pa_protocol'] = pa

            wrapper = ProtocolWrapperServerClient(
                protocol=proto,
                pa_protocol=pa,
                num_qubits=num_qubits,
                freq=cfg.get('freq', 1e7),
                mu=cfg.get('mu', 1.0),
                att_db_km=cfg.get('att_db_km', 0.2),
                detector_eff=cfg.get('det_eff', 0.8),
                dark_count=cfg.get('dark_count', 0.01),
                protocol_params=cfg.get('protocol_params', {}),
                verbose=False,
                seeds=cfg.get('seeds', {}),
            )
            wrapper.num_trials = int(cfg.get('num_trials', 3))

            chain = MultiHopChain(**chain_params)
            relayed_rates = []

            for total_dist in scan_distances:
                per_link_distances = [max(total_dist * ratio, 0.1) for ratio in segment_ratios]
                per_link_rates = []
                for dist_km in per_link_distances:
                    rate_bps, _ = wrapper.calculate_rate(dist_km)
                    per_link_rates.append(rate_bps)
                relayed_rates.append(
                    chain.calculate_chain_rate(
                        per_link_rates,
                        per_link_distances,
                        [link_capacity] * len(per_link_distances),
                    )
                )

            results[f'{proto}/{pa}'] = (scan_distances.tolist(), relayed_rates)

    return scan_distances, results, max(0, len(usable_chain) - 1)


def build_multihop_serverclient_dashboard(
    chain_data,
    selected_ec_protocols=None,
    selected_pa_protocols=None,
    num_qubits=5000,
    link_capacity=1e9,
    sim_config=None,
    chain_params=None,
    relay_sweep_max=40,
    relay_sweep_points=20,
    max_distance=None,
    num_points=20,
):
    if sim_config is None:
        sim_config = {}
    if chain_params is None:
        chain_params = {}

    usable_chain = chain_data or [{'distance': 50.0, 'capacity': link_capacity}]
    total_distance = sum(float(seg.get('distance', 0.0)) for seg in usable_chain)
    if max_distance is None:
        max_distance = max(total_distance, 200.0)

    relay_counts = np.unique(np.linspace(1, relay_sweep_max, relay_sweep_points, dtype=int)).tolist()
    if not relay_counts:
        relay_counts = [1]

    range_protocols = selected_ec_protocols or [sim_config.get('protocol', 'polar')]
    range_pas = selected_pa_protocols or [sim_config.get('pa_protocol', 'toeplitz')]

    relay_counts, hop_results = simulate_optimal_hops(
        protocols=range_protocols,
        pa_protocols=range_pas,
        total_distance=total_distance,
        relay_counts=relay_counts,
        num_qubits=num_qubits,
        link_capacity=link_capacity,
        sim_config=sim_config,
        chain_params=chain_params,
    )
    
    effective_rate = 0.0
    for rates in hop_results.values():
        if rates and max(rates) > effective_rate:
            effective_rate = float(max(rates))

    optimal_fig = create_optimal_hops_figure(relay_counts, hop_results, total_distance)
    plot_optimal_hops_base64 = _figure_to_base64(optimal_fig)
    plt.close(optimal_fig)

    scan_distances, range_results, relays = simulate_range_extension_protocol_pairs(
        chain_data=usable_chain,
        protocols=range_protocols,
        pa_protocols=range_pas,
        max_distance=max_distance,
        num_points=num_points,
        num_qubits=num_qubits,
        link_capacity=link_capacity,
        sim_config=sim_config,
        chain_params=chain_params,
    )

    range_fig = create_range_extension_protocols_figure(scan_distances, range_results, relays)
    plot_range_extension_base64 = _figure_to_base64(range_fig)
    plt.close(range_fig)

    return {
        'submitted': True,
        'effective_rate': effective_rate,
        'total_distance': total_distance,
        'plot_optimal_hops': plot_optimal_hops_base64,
        'plot_range_extension': plot_range_extension_base64,
        'relay_counts': relay_counts,
        'hop_results': hop_results,
        'range_distances': scan_distances.tolist(),
        'range_results': range_results,
        'range_relays': relays,
    }


def plot_optimal_hops(x_data, results, total_dist, title=None):
    plt.figure(figsize=(12, 7))
    colors = plt.cm.get_cmap('tab10')

    if title is None:
        title = f'Optimal Hop Analysis — Protocol Comparison\nTotal Dist={total_dist}km'

    all_positive_vals = []

    for i, (combo_name, y_data) in enumerate(results.items()):
        color = colors(i % 10)
        y_arr = np.asarray(y_data, dtype=float)
        plt.plot(x_data, y_arr, 'o-', color=color, linewidth=2, label=combo_name)
        all_positive_vals.append(y_arr[y_arr > 0])

    plt.xlabel('Number of Intermediate Relays')
    plt.ylabel('Effective End-to-End Rate (bps)')
    plt.title(title)
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.yscale('log')
    plt.legend(fontsize='small', ncol=2)
    
    try:
        positives = np.hstack([a for a in all_positive_vals if a.size > 0]) if all_positive_vals else np.array([])
        if positives.size > 0:
            y_min_pos = positives.min()
            plt.ylim(bottom=max(y_min_pos * 0.1, 1e-12))
    except Exception:
        pass
    
    plt.show()


def plot_range_extension(distances, direct_rates, relayed_rates, relays, config):
    plt.figure(figsize=(10, 6))
    p_name = config.get('protocol', 'polar').upper()
    pa_name = config.get('pa_protocol', 'toeplitz').upper()
    plt.plot(distances, direct_rates, 'k--', label='Direct Link')
    plt.plot(distances, relayed_rates, 'g-', linewidth=2, label=f'Chain with {relays} Relays')
    plt.xlabel('Total Distance Alice-Bob (km)')
    plt.ylabel('Effective Key Rate (bps)')
    plt.title(f'Range Extension: Direct vs Relayed\n(Server-Client BB84 {p_name}/{pa_name})')
    plt.legend()
    plt.yscale('log')
    plt.ylim(bottom=1)
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.show()


def run_multihop_serverclient_example():
    SIM_CONFIG = {
        'freq': 1e7,
        'mu': 0.1,
        'att_db_km': 0.2,
        'det_eff': 0.8,
        'dark_count': 0.001,
        'protocol_params': {'u_fer_target': 0.01}
    }

    CHAIN_CONFIG = {'sifting_exchanges': 3, 'overhead_factor': 3, 'packet_size': 10000}

    print('--- Server-Client Multihop Simulation Example ---')
    
    protocols = ['polar', 'cascade', 'nr_ldpc_standard', 'ldpc_rateadaptive']
    pa_protocols = ['toeplitz']

    # Optimal hops
    relays, hop_results = simulate_optimal_hops(
        protocols=protocols, pa_protocols=pa_protocols, total_distance=500.0, 
        relay_counts=np.linspace(1, 30, 20, dtype=int).tolist(), num_qubits=10000, 
        link_capacity=1e9, sim_config=SIM_CONFIG, chain_params=CHAIN_CONFIG
    )
    plot_optimal_hops(relays, hop_results, 500.0)

    # Range extension
    dists, results = simulate_range_extension_protocols(
        protocols=protocols, pa_protocols=pa_protocols, max_distance=700, 
        num_points=20, comparison_relays=3, num_qubits=10000, 
        link_capacity=1e9, sim_config=SIM_CONFIG, chain_params=CHAIN_CONFIG
    )
    plot_range_extension_protocols(dists, results, 3, SIM_CONFIG)


if __name__ == '__main__':
    run_multihop_serverclient_example()