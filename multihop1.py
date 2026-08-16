import numpy as np
import matplotlib.pyplot as plt

# Import both protocol definitions
import bb84_finite as bb84
import DPS_finite as dps

class ProtocolWrapper:
    """
    Unified interface for different QKD protocols (BB84 and DPS).
    Abstracts the differences in initialization and SKR calculation.
    """
    def __init__(self, protocol_type='bb84', channel_mode='fiber', 
                 freq=1e7, n_em=1e13, att_db_km=0.2):
        self.protocol_type = protocol_type.lower()
        self.channel_mode = channel_mode.lower()
        self.freq = freq
        self.n_em = n_em # Used for DPS block size
        self.att_db_km = att_db_km

        # --- Initialize Specific Protocol ---
        if self.protocol_type == 'bb84':
            self.source = bb84.Source(freq=self.freq, q=1.0, alignment_error=0.01)
            self.detector = bb84.Detector(efficiency=0.1, dark_count_rate=1e5)
            self.channel = bb84.Channel(att_db_km=self.att_db_km, channel_mode=self.channel_mode)
            self.protocol = bb84.Protocol(self.source, self.channel, self.detector)
            
            
        elif self.protocol_type == 'dps':
            self.source = dps.IdealSinglePhotonSource()
            self.detector = dps.Detector(dark_count_rate=1e-6, efficiency=0.1)
            # Note: DPS_finite's Channel sets alpha internally usually, 
            # but we can try to match parameters if the class allows, 
            # otherwise it defaults to standard fiber/fso params.
            self.channel = dps.Channel(mode=self.channel_mode)
            # DPS uses n_em (block size) and rep_rate (freq) in init
            self.protocol = dps.DPSProtocol(
                self.source, self.channel, self.detector, 
                n_em=self.n_em, rep_rate=freq
            )
        else:
            raise ValueError(f"Unknown protocol type: {protocol_type}")

    def set_distance(self, distance_km):
        """Sets the distance for the underlying channel."""
        # Both channel classes support updating L or using set_distance
        if self.protocol_type == 'bb84':
            self.protocol.channel.L = distance_km
        elif self.protocol_type == 'dps':
            self.protocol.channel.set_distance(distance_km)

    def calculate_rate(self, block_size_N=None):
        """
        Calculates Secret Key Rate (bps).
        
        Args:
            block_size_N: Only used for BB84 (finite block size). 
                          DPS uses 'n_em' defined at init.
        """
        if self.protocol_type == 'bb84':
            # BB84 requires an explicit block size N for every calc
            N = block_size_N if block_size_N else 1e8
            rate, _ = self.protocol.calculate_skr(N)
            return rate
            
        elif self.protocol_type == 'dps':
            # DPS uses differential_evolution optimization
            return dps.optimize_skr(self.protocol)
        
        return 0.0

class MultiHopChain:
    def __init__(self, sifting_exchanges=3, overhead_factor=3, packet_size=10000):
        self.S = sifting_exchanges    # QKD Sifting exchanges per hop
        self.n = overhead_factor      # QKD Classical overhead factor
        self.c = 3e5                  # Speed of light in km/s
        self.L_packet = packet_size   # Consistent block size

    def calculate_chain_rate(self, link_rates, link_distances, link_capacities):
        """
        Calculates effective rate for a chain using Parallel QKD + Sequential Forwarding logic.
        """
        # 1. Check for broken links (cutoff at 1e-9 bps)
        if not link_rates or any(k <= 1e-9 for k in link_rates):
            return 0.0

        # THROUGHPUT (Steady State):
        # keys the entire system can produce is limited by the slowest link.
        bottleneck_rate = min(link_rates) # bps
        
        # 1. Generation Time (limited by slowest link)
        t_generation_bottleneck = (2 * self.L_packet) / bottleneck_rate

        # 2. Latency (Accumulative)
        latency_total = 0
        
        for i in range(len(link_rates)):
            D = link_distances[i]
            C = link_capacities[i]
            
            # Propagation delay
            t_prop = (self.S * D) / self.c
            
            # Transmission delay
            t_trans = (self.n * self.L_packet) / C
            
            latency_total += t_prop + t_trans

        # Total Time per Packet Cycle
        total_tau = t_generation_bottleneck + latency_total
        
        # Effective Rate = Bits / Time
        return self.L_packet / total_tau

# ==========================================
# SIMULATION FUNCTIONS
# ==========================================

def simulate_optimal_hops(
    total_distance=300.0,
    relay_counts=None,
    block_size_N=1e8,         # Relevant for BB84
    link_capacity=1e9,
    simulation_config=None,   # Contains protocol_type, channel_mode, etc.
    chain_params=None
):
    if relay_counts is None:
        relay_counts = [1, 2, 4, 9, 14, 19, 29, 49]
    if chain_params is None: chain_params = {}
    if simulation_config is None: simulation_config = {}

    # Initialize Wrapper
    # Extract config to pass to wrapper
    p_type = simulation_config.get('protocol', 'bb84')
    c_mode = simulation_config.get('channel', 'fiber')
    freq = simulation_config.get('freq', 1e7)
    
    # BB84 usually works on N (sifted block), DPS works on N_em (emitted block)
    # We pass N_em for DPS here if needed
    n_em = simulation_config.get('dps_n_em', 1e13)

    wrapper = ProtocolWrapper(
        protocol_type=p_type, 
        channel_mode=c_mode, 
        freq=freq,
        n_em=n_em
    )
    
    chain_logic = MultiHopChain(**chain_params)
    final_rates = []

    print(f"   -> Simulating {p_type.upper()} over {c_mode}...")

    for num_relays in relay_counts:
        num_links = num_relays + 1
        dist_per_link = total_distance / num_links
        
        # Calculate raw physics for one segment
        wrapper.set_distance(dist_per_link)
        segment_rate = wrapper.calculate_rate(block_size_N=block_size_N)
        
        if segment_rate <= 1e-9: # Cutoff for dead keys
            final_rates.append(0)
            continue
            
        # Build lists for the chain function
        rates = [segment_rate] * num_links
        dists = [dist_per_link] * num_links
        caps  = [link_capacity] * num_links 
        
        eff_rate = chain_logic.calculate_chain_rate(rates, dists, caps)
        final_rates.append(eff_rate)
        
    return relay_counts, final_rates

def simulate_range_extension(
    max_distance=700,
    num_points=50,
    comparison_relays=3,
    block_size_N=1e8,
    link_capacity=1e9,
    simulation_config=None,
    chain_params=None
):
    if chain_params is None: chain_params = {}
    if simulation_config is None: simulation_config = {}

    # Initialize Wrapper
    p_type = simulation_config.get('protocol', 'bb84')
    c_mode = simulation_config.get('channel', 'fiber')
    freq = simulation_config.get('freq', 1e7)
    n_em = simulation_config.get('dps_n_em', 1e13)

    wrapper = ProtocolWrapper(
        protocol_type=p_type, 
        channel_mode=c_mode, 
        freq=freq, 
        n_em=n_em
    )
    
    chain_logic = MultiHopChain(**chain_params)

    scan_distances = np.linspace(10, max_distance, num_points)
    rates_direct = []
    rates_relayed = []
    
    num_links_relayed = comparison_relays + 1
    
    print(f"   -> Simulating Range Extension ({p_type.upper()}/{c_mode})...")

    for D in scan_distances:
        # --- Scenario A: Direct ---
        wrapper.set_distance(D)
        k_direct = wrapper.calculate_rate(block_size_N=block_size_N)
        
        r_direct = chain_logic.calculate_chain_rate(
            [k_direct], [D], [link_capacity]
        )
        rates_direct.append(r_direct)
        
        # --- Scenario B: Chain ---
        d_seg = D / num_links_relayed
        wrapper.set_distance(d_seg)
        k_seg = wrapper.calculate_rate(block_size_N=block_size_N)
        
        r_chain = chain_logic.calculate_chain_rate(
            [k_seg] * num_links_relayed, 
            [d_seg] * num_links_relayed, 
            [link_capacity] * num_links_relayed
        )
        rates_relayed.append(r_chain)
        
    return scan_distances, rates_direct, rates_relayed

# ==========================================
# PLOTTING FUNCTIONS
# ==========================================

def plot_optimal_hops(x_data, y_data, total_dist, config, title=None):
    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, 'bo-', linewidth=2)
    
    p_name = config.get('protocol', 'bb84').upper()
    c_name = config.get('channel', 'fiber').upper()
    
    if title is None:
        title = f'Optimal Hop Analysis ({p_name} over {c_name})\nTotal Dist={total_dist}km'
        
    plt.xlabel('Number of Intermediate Relays')
    plt.ylabel('Effective End-to-End Rate (bps)')
    plt.title(title)
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.yscale('log')
    plt.show()

def plot_range_extension(distances, direct_rates, relayed_rates, relays, config):
    plt.figure(figsize=(10, 6))
    
    p_name = config.get('protocol', 'bb84').upper()
    c_name = config.get('channel', 'fiber').upper()
    
    plt.plot(distances, direct_rates, 'k--', label='Direct Link')
    plt.plot(distances, relayed_rates, 'g-', linewidth=2, label=f'Chain with {relays} Relays')
    
    plt.xlabel('Total Distance Alice-Bob (km)')
    plt.ylabel('Effective Key Rate (bps)')
    plt.title(f'Range Extension: Direct vs Relayed\n({p_name} over {c_name})')
    plt.legend()
    plt.yscale('log')
    plt.ylim(bottom=1) 
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.show()

# ==========================================
# MAIN EXECUTION
# ==========================================

def run_multihop_analysis():
    # ====================================================
    # USER CONFIGURATION SECTION
    # ====================================================
    
    # 1. Choose Protocol: 'bb84' or 'dps'
    SELECTED_PROTOCOL = 'bb84' 
    
    # 2. Choose Channel: 'fiber' or 'fso'
    SELECTED_CHANNEL = 'fiber'
    
    # 3. Physics Parameters
    PHYSICS_CONFIG = {
        'protocol': SELECTED_PROTOCOL,
        'channel': SELECTED_CHANNEL,
        'freq': 1e7,          # Pulse repetition rate (10 MHz)
        'dps_n_em': 1e13      # Only used if protocol is DPS
    }
    
    # 4. Network Parameters
    CHAIN_CONFIG = {
        'sifting_exchanges': 3,
        'overhead_factor': 3,
        'packet_size': 10000
    }
    
    BLOCK_SIZE_BB84 = 1e8     # Only used if protocol is BB84
    CLASSICAL_CAPACITY = 1e9  # 1 Gbps

    print(f"--- Starting Simulation: {SELECTED_PROTOCOL.upper()} over {SELECTED_CHANNEL.upper()} ---")

    # ------------------------------------------
    # SIMULATION 1: Optimal Hop Count
    # ------------------------------------------
    print("\n[1/2] Calculating Optimal Hop Count...")
    target_distance = 300.0 # Total distance end-to-end
    
    relays, eff_rates = simulate_optimal_hops(
        total_distance=target_distance,
        block_size_N=BLOCK_SIZE_BB84,
        link_capacity=CLASSICAL_CAPACITY,
        simulation_config=PHYSICS_CONFIG,
        chain_params=CHAIN_CONFIG
    )
    
    plot_optimal_hops(relays, eff_rates, target_distance, PHYSICS_CONFIG)

    # ------------------------------------------
    # SIMULATION 2: Range Extension
    # ------------------------------------------
    print("\n[2/2] Calculating Range Extension...")
    relay_comparison_count = 3 
    max_scan_dist = 600.0
    
    dists, r_direct, r_relayed = simulate_range_extension(
        max_distance=max_scan_dist,
        comparison_relays=relay_comparison_count,
        block_size_N=BLOCK_SIZE_BB84,
        link_capacity=CLASSICAL_CAPACITY,
        simulation_config=PHYSICS_CONFIG,
        chain_params=CHAIN_CONFIG
    )
    
    plot_range_extension(dists, r_direct, r_relayed, relay_comparison_count, PHYSICS_CONFIG)

if __name__ == "__main__":
    run_multihop_analysis()