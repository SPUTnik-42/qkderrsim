import asyncio
import random
import logging
import datetime
import math
import matplotlib.pyplot as plt
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Tuple, Any
from abc import ABC, abstractmethod

# --- Logging Configuration ---
logging.basicConfig(level=logging.ERROR, format='%(message)s')

# --- Primitives ---

class Basis(Enum):
    RECTILINEAR = "+" 
    DIAGONAL = "x"   

@dataclass
class Photon:
    id: int
    bit: int
    basis: Basis
    creation_time: str
    wavelength_nm: float = 1550.0

@dataclass
class DetectionEvent:
    photon_id: int
    basis: Basis
    bit: int
    detection_time: str
    source: str 

# --- CENTRAL PRNG CLASS ---

class PRNG:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PRNG, cls).__new__(cls)
        return cls._instance

    def seed(self, val):
        random.seed(val)

    def get_bit(self) -> int: return random.choice([0, 1])
    def get_basis(self) -> Basis: return random.choice(list(Basis))
    def random(self) -> float: return random.random()
    def shuffle(self, x: list): random.shuffle(x)
    def choice(self, options: list): return random.choice(options)

prng = PRNG()

# --- HELPER: Table Printer ---

def print_key_comparison_table(alice_key, bob_key_raw, bob_key_corrected, indices_to_show=20):
    """Prints a side-by-side comparison of keys."""
    n = len(alice_key)
    limit = min(n, indices_to_show)
    
    print("\n--- KEY COMPARISON (First {} bits) ---".format(limit))
    print(f"{'Idx':<5} | {'Alice':<5} | {'Bob(Raw)':<8} | {'Bob(Final)':<10} | {'Status'}")
    print("-" * 55)
    
    for i in range(limit):
        a = alice_key[i]
        b_raw = bob_key_raw[i]
        b_final = bob_key_corrected[i]
        
        status = "OK"
        if a != b_raw and a == b_final: status = "FIXED"
        elif a != b_raw and a != b_final: status = "ERROR"
        elif a == b_raw and a != b_final: status = "BROKE" 
        
        print(f"{i:<5} | {a:<5} | {b_raw:<8} | {b_final:<10} | {status}")
    
    if n > limit:
        print(f"... ({n - limit} more bits omitted) ...")
    print("-" * 55 + "\n")

# --- ERROR CORRECTION MODULES ---

class BaseErrorCorrection(ABC):
    @abstractmethod
    def reconcile(self, alice_key: List[int], bob_key: List[int], verbose: bool = False) -> Tuple[List[int], int, int]:
        pass

class Cascade(BaseErrorCorrection):
    def __init__(self, num_passes=4, initial_block_size=8):
        self.num_passes = num_passes
        self.initial_block_size = initial_block_size

    def _get_parity(self, key_segment):
        return sum(key_segment) % 2

    def reconcile(self, alice_key: List[int], bob_key: List[int], verbose: bool = False) -> Tuple[List[int], int, int]:
        if verbose: print(f"\n[CASCADE] Starting Reconciliation (Passes={self.num_passes}, BlockSize={self.initial_block_size})")
        
        bob_corrected = list(bob_key)
        n = len(alice_key)
        if n == 0: return [], 0, 0
        
        bits_revealed = 0
        total_errors_corrected = 0
        indices = list(range(n))
        current_block_size = self.initial_block_size

        for pass_idx in range(self.num_passes):
            if verbose: print(f"  > Pass {pass_idx + 1}/{self.num_passes} (Block Size: {current_block_size})")
            
            if pass_idx > 0:
                prng.shuffle(indices)
                current_block_size = min(n, current_block_size * 2)

            blocks = [indices[i:i + current_block_size] for i in range(0, n, current_block_size)]
            pass_errors = 0
            
            for block_num, block_indices in enumerate(blocks):
                alice_vals = [alice_key[i] for i in block_indices]
                bob_vals = [bob_corrected[i] for i in block_indices]
                
                p_alice = self._get_parity(alice_vals)
                p_bob = self._get_parity(bob_vals)
                bits_revealed += 1
                
                if p_alice != p_bob:
                    if verbose: print(f"    - Parity Mismatch at Block {block_num}. Starting BINARY SEARCH...")
                    corrected_idx, cost = self._binary_search_and_correct(
                        alice_key, bob_corrected, block_indices
                    )
                    bits_revealed += cost
                    total_errors_corrected += 1
                    pass_errors += 1
                    if verbose: print(f"      -> Fixed Bit Index {corrected_idx} (Cost: {cost} bits)")

            if verbose: print(f"  > Pass {pass_idx+1} Complete. Errors found: {pass_errors}")

        return bob_corrected, bits_revealed, total_errors_corrected

    def _binary_search_and_correct(self, alice_full, bob_full, indices):
        cost = 0
        current_indices = indices

        while len(current_indices) > 1:
            split = len(current_indices) // 2
            left_indices = current_indices[:split]
            
            a_left = [alice_full[i] for i in left_indices]
            b_left = [bob_full[i] for i in left_indices]
            
            p_a = self._get_parity(a_left)
            p_b = self._get_parity(b_left)
            cost += 1
            
            if p_a != p_b:
                current_indices = left_indices
            else:
                current_indices = current_indices[split:]
        
        err_idx = current_indices[0]
        bob_full[err_idx] = 1 - bob_full[err_idx]
        return err_idx, cost

# --- Actor Framework ---

class Actor:
    def __init__(self, name):
        self.name = name
        self.mailbox = asyncio.Queue()
        self._running = False

    async def start(self):
        self._running = True
        while self._running:
            msg = await self.mailbox.get()
            if isinstance(msg, tuple) and msg[0] == "STOP":
                self._running = False
                break
            await self.handle_message(msg)

    async def send(self, recipient: 'Actor', msg):
        await recipient.mailbox.put(msg)

    async def handle_message(self, msg): pass

# --- Physical Actors ---

class QuantumChannel(Actor):
    def __init__(self, name, length_km, attenuation_db, optical_error_rate, next_actor):
        super().__init__(name)
        self.length = length_km
        self.att_db = attenuation_db
        self.optical_error_rate = optical_error_rate
        self.next_actor = next_actor
        self.transmittance = 10**(-(length_km * attenuation_db)/10)

    async def handle_message(self, msg):
        if not isinstance(msg, Photon):
            await self.send(self.next_actor, msg)
            return

        if prng.random() > self.transmittance: return # Photon lost

        final_bit = msg.bit
        # Simulate optical errors (e.g. polarization drift)
        if prng.random() < self.optical_error_rate:
            final_bit = 1 - final_bit 

        out_photon = Photon(msg.id, final_bit, msg.basis, msg.creation_time)
        await self.send(self.next_actor, out_photon)

class Detector(Actor):
    def __init__(self, name, efficiency, dark_count_prob, parent_bob):
        super().__init__(name)
        self.eta = efficiency
        self.p_dc = dark_count_prob
        self.parent = parent_bob

    async def handle_message(self, msg):
        current_time = datetime.datetime.now().strftime("%H:%M:%S.%f")
        click = False
        measured_bit = -1
        bob_basis = None
        source = "None"

        # 1. Dark Count check
        if prng.random() < self.p_dc:
            click = True
            source = "DarkCount"
            measured_bit = prng.get_bit()
            bob_basis = prng.get_basis()

        # 2. Photon detection check
        if not click and isinstance(msg, Photon):
            if prng.random() < self.eta:
                click = True
                source = "Signal"
                bob_basis = prng.get_basis()
                if bob_basis == msg.basis:
                    measured_bit = msg.bit
                else:
                    measured_bit = prng.get_bit()
                    source = "Signal(BasisMismatch)"

        if click:
            event = DetectionEvent(
                photon_id=msg.id if isinstance(msg, Photon) else -1,
                basis=bob_basis,
                bit=measured_bit,
                detection_time=current_time,
                source=source
            )
            await self.send(self.parent, event)

class Eve(Actor):
    def __init__(self, name, next_actor):
        super().__init__(name)
        self.next_actor = next_actor

    async def handle_message(self, msg):
        if not isinstance(msg, Photon): return
        
        eve_basis = prng.get_basis()
        measured_bit = msg.bit
        
        if eve_basis != msg.basis:
            measured_bit = prng.get_bit()
            
        new_photon = Photon(msg.id, measured_bit, eve_basis, msg.creation_time)
        await self.send(self.next_actor, new_photon)

# --- Protocol Agents ---

class Alice(Actor):
    def __init__(self, name, channel, num_qubits, verbose=False):
        super().__init__(name)
        self.channel = channel
        self.num_qubits = num_qubits
        self.sent_log = {}
        self.verbose = verbose

    async def run_protocol(self):
        if self.verbose: print(f"[{self.name}] Generating and sending {self.num_qubits} qubits...")
        for i in range(self.num_qubits):
            bit = prng.get_bit()
            basis = prng.get_basis()
            t_now = datetime.datetime.now().strftime("%H:%M:%S.%f")
            p = Photon(id=i, bit=bit, basis=basis, creation_time=t_now)
            self.sent_log[i] = p
            await self.send(self.channel, p)
            if i % 1000 == 0: await asyncio.sleep(0.0001) # Yield to event loop
        if self.verbose: print(f"[{self.name}] Transmission complete.")

class Bob(Actor):
    def __init__(self, name):
        super().__init__(name)
        self.received_log = []

    async def handle_message(self, msg):
        if isinstance(msg, DetectionEvent):
            self.received_log.append(msg)

# --- Post Processing & Sifting ---

def sift_keys(alice_sent_log, bob_received_log, verbose=False) -> Tuple[List[int], List[int]]:
    if verbose: print(f"\n[SIFTING] Alice sent {len(alice_sent_log)}, Bob detected {len(bob_received_log)}")
    sifted_alice = []
    sifted_bob = []
    
    match_count = 0
    basis_mismatch_count = 0
    
    for rx in bob_received_log:
        if rx.photon_id not in alice_sent_log: continue
        tx = alice_sent_log[rx.photon_id]
        
        if tx.basis == rx.basis:
            sifted_alice.append(tx.bit)
            sifted_bob.append(rx.bit)
            match_count += 1
        else:
            basis_mismatch_count += 1
            
    if verbose: 
        print(f"[SIFTING] Bases Matched: {match_count}")
        print(f"[SIFTING] Bases Mismatched (discarded): {basis_mismatch_count}")
        
    return sifted_alice, sifted_bob

def calculate_metrics(reference_key, subject_key, num_input_qubits):
    n_sifted = len(reference_key)
    if n_sifted == 0:
        return {"qber": 0.0, "yield": 0.0, "length": 0, "errors": 0}

    errors = sum(1 for a, b in zip(reference_key, subject_key) if a != b)
    qber = errors / n_sifted
    yield_rate = n_sifted / num_input_qubits if num_input_qubits > 0 else 0.0
    
    return {
        "qber": qber,
        "yield": yield_rate,
        "length": n_sifted,
        "errors": errors
    }

# --- Core Simulation Wrapper ---

async def run_simulation_instance(num_qubits=100, include_eve=False, verbose=True, error_correction=None, optical_error_rate=0.02):
    if verbose:
        print("="*60)
        print(f"STARTING SIMULATION: {num_qubits} Qubits | Eve={include_eve}")
        print("="*60)
    prng.seed(42)
    # Parameters
    LENGTH_KM = 50       
    ATTENUATION = 0.2 
    OPTICAL_ERROR = optical_error_rate # Use parameter instead of hardcoded value
    DETECTOR_EFF = 0.8    
    DARK_COUNT = 0.01    

    bob = Bob("Bob")
    detector = Detector("BobSPD", DETECTOR_EFF, DARK_COUNT, bob)
    
    if include_eve:
        eve = Eve("Eve", next_actor=detector)
        channel = QuantumChannel("Fiber", LENGTH_KM, ATTENUATION, OPTICAL_ERROR, next_actor=eve)
        alice = Alice("Alice", channel, num_qubits, verbose=verbose)
        actors = [alice, channel, eve, detector, bob]
    else:
        channel = QuantumChannel("Fiber", LENGTH_KM, ATTENUATION, OPTICAL_ERROR, next_actor=detector)
        alice = Alice("Alice", channel, num_qubits, verbose=verbose)
        actors = [alice, channel, detector, bob]

    tasks = [asyncio.create_task(a.start()) for a in actors]

    await alice.run_protocol()
    
    # Wait for flush
    await asyncio.sleep(0.1) 
    for a in actors: await a.send(a, ("STOP",))
    await asyncio.gather(*tasks)
    
    # 1. Sifting
    sift_alice, sift_bob = sift_keys(alice.sent_log, bob.received_log, verbose=verbose)
    
    # Calculate Raw Metrics
    raw_metrics = calculate_metrics(sift_alice, sift_bob, num_qubits)
    
    if verbose:
        print(f"\n[METRICS - RAW]")
        print(f"  Sifted Length : {raw_metrics['length']}")
        print(f"  Bit Errors    : {raw_metrics['errors']}")
        print(f"  QBER (Raw)    : {raw_metrics['qber']:.4%}")

    final_key_bob = list(sift_bob)
    ec_stats = {"corrected": 0, "revealed": 0}

    # 2. Error Correction
    if error_correction and len(sift_alice) > 0:
        final_key_bob, revealed, corrected = error_correction.reconcile(sift_alice, sift_bob, verbose=verbose)
        ec_stats["corrected"] = corrected
        ec_stats["revealed"] = revealed

    # Calculate Final Metrics
    final_metrics = calculate_metrics(sift_alice, final_key_bob, num_qubits)

    if verbose:
        print(f"\n[METRICS - FINAL]")
        print(f"  Final Errors  : {final_metrics['errors']} (Should be 0)")
        print(f"  QBER (Final)  : {final_metrics['qber']:.4%}")
        print(f"  Bits Leaked   : {ec_stats['revealed']}")
        
        # Print comparison table
        print_key_comparison_table(sift_alice, sift_bob, final_key_bob, indices_to_show=25)

    return {
        "raw_qber": raw_metrics["qber"],
        "final_qber": final_metrics["qber"],
        "sifted_length": raw_metrics["length"],
        "ec_revealed_bits": ec_stats["revealed"],
        "ec_corrected_errors": ec_stats["corrected"],
        "yield": raw_metrics["yield"]
    }

# --- Plotting Function ---

async def run_plotting_experiment():
    print("\n" + "="*80)
    print("STARTING EXTENDED PLOTTING EXPERIMENT")
    print("="*80)
    prng.seed(42)
    
    qubit_counts = [100, 500, 1000, 5000, 10000, 20000, 50000,100000, 200000, 500000, 1000000]
    cascade_engine = Cascade(num_passes=10, initial_block_size=5)
    
    # Storage for data
    data_safe = {"counts": [], "qber": [],"final_qber": [], "yield": [], "leaked_ratio": []}
    data_attacked = {"counts": [], "qber": [], "final_qber": [], "yield": [], "leaked_ratio": []}

    # --- Run 1: Without Eve (Safe) ---
    print("\nRunning Batch 1: No Eavesdropper...")
    for q_count in qubit_counts:
        res = await run_simulation_instance(q_count, include_eve=False, verbose=False, error_correction=cascade_engine)
        data_safe["counts"].append(q_count)
        data_safe["qber"].append(res['raw_qber'] * 100)
        data_safe["final_qber"].append(res['final_qber'] * 100)
        data_safe["yield"].append(res['sifted_length']) # Using length for the yield graph
        # Calculate leakage ratio: bits revealed / sifted key length
        ratio = res['ec_revealed_bits'] / res['sifted_length'] if res['sifted_length'] > 0 else 0
        data_safe["leaked_ratio"].append(ratio * 100)

    # --- Run 2: With Eve (Attacked) ---
    print("Running Batch 2: With Eve...")
    for q_count in qubit_counts:
        res = await run_simulation_instance(q_count, include_eve=True, verbose=False, error_correction=cascade_engine)
        data_attacked["counts"].append(q_count)
        data_attacked["qber"].append(res['raw_qber'] * 100)
        data_attacked["final_qber"].append(res['final_qber'] * 100)
        data_attacked["yield"].append(res['sifted_length'])
        ratio = res['ec_revealed_bits'] / res['sifted_length'] if res['sifted_length'] > 0 else 0
        data_attacked["leaked_ratio"].append(ratio * 100)

    # --- PLOTTING ---
    
    # Graph 1: The "Eve Effect" (QBER Comparison)
    plt.figure(figsize=(10, 6))
    plt.plot(data_safe["counts"], data_safe["qber"], 'o-', color='green', label='No Eve (Safe)')
    plt.plot(data_attacked["counts"], data_attacked["qber"], 'x--', color='red', label='With Eve (Attacked)')
    plt.xscale('log')
    plt.xlabel('Input Qubit Count')
    plt.ylabel('Raw Quantum Bit Error Rate (QBER) %')
    plt.title('Impact of Eavesdropping on Error Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Graph 1.5: The "EC Effect" (Error Corrected QBER Comparison)
    plt.figure(figsize=(10, 6))
    plt.plot(data_attacked["counts"], data_attacked["final_qber"], 'o-', color='green', label='With Eve (Error Corrected)')
    plt.plot(data_attacked["counts"], data_attacked["qber"], 'x--', color='red', label='With Eve (Raw)')
    plt.xscale('log')
    plt.xlabel('Input Qubit Count')
    plt.ylabel('Quantum Bit Error Rate (QBER) %')
    plt.title('Impact of Eavesdropping on Error Rate with and without error correction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Graph 2: The "Funnel of Loss" (Key Yield)
    plt.figure(figsize=(10, 6))
    plt.plot(qubit_counts, qubit_counts, '--', color='gray', label='Input Qubits')
    plt.plot(data_safe["counts"], data_safe["yield"], 'o-', color='blue', label='Sifted Key (After Loss)')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Input Qubits Generated')
    plt.ylabel('Resulting Key Length (Bits)')
    plt.title('System Throughput: Input vs. Sifted Key')
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.show()

    # Graph 3: Cost of Error Correction (Leakage)
    plt.figure(figsize=(10, 6))
    plt.plot(data_safe["counts"], data_safe["leaked_ratio"], 's-', color='green', label='Safe (Low Error)')
    plt.plot(data_attacked["counts"], data_attacked["leaked_ratio"], 'd--', color='red', label='Attacked (High Error)')
    plt.xscale('log')
    plt.xlabel('Input Qubit Count')
    plt.ylabel('Key Leakage (% of Sifted Key Revealed)')
    plt.title('Information Leakage during Cascade Error Correction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# --- NEW: Efficiency vs QBER Experiment ---

def binary_entropy(p):
    """Calculates binary entropy H(p)."""
    if p <= 0 or p >= 1: return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

async def run_efficiency_experiment():
    print("\n" + "="*80)
    print("STARTING RECONCILIATION EFFICIENCY vs QBER EXPERIMENT")
    print("="*80)
    
    # Parameters for the continuous graph
    start_err = 0.012
    end_err = 0.160
    num_points = 60
    
    # Generate finer-grained error rates to create a continuous look
    step_size = (end_err - start_err) / num_points
    error_rates = [start_err + i * step_size for i in range(num_points + 1)]
    
    efficiencies = []
    observed_qbers = []
    
    # Use a fixed block size that works reasonably well
    cascade_engine = Cascade(num_passes=10, initial_block_size=8)
    
    for err in error_rates:
        # Run simulation with 10k qubits to get stable statistics
        res = await run_simulation_instance(
            num_qubits=10000, 
            include_eve=False, 
            verbose=False, 
            error_correction=cascade_engine,
            optical_error_rate=err
        )
        
        N = res['ec_revealed_bits']  # Total bits revealed
        K = res['sifted_length']     # Key length (sifted)
        eps = res['raw_qber']        # QBER
        
        # Calculate Efficiency η = N / (K * H(ε))
        if K > 0 and eps > 0 and eps < 1:
            h_eps = binary_entropy(eps)
            if h_eps > 0:
                eff = N / (K * h_eps)
                # Filter outliers or division artifacts near 0
                if eff > 0.5 and eff < 10: 
                    efficiencies.append(eff)
                    observed_qbers.append(eps)
                    print(f"OpticalErr: {err:.4f} | QBER: {eps:.4f} | Leaked: {N}/{K} | Eff: {eff:.4f}")

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    
    # Sort data for clean line plotting (just in case of async/random jitter)
    sorted_pairs = sorted(zip(observed_qbers, efficiencies))
    x_vals = [p[0] for p in sorted_pairs]
    y_vals = [p[1] for p in sorted_pairs]

    plt.plot(x_vals, y_vals, 'o-', color='purple', markersize=4, label='Cascade Efficiency (Simulated)')
    
    # Setup Axes limits as requested
    plt.xlim(0, 0.18)       # X axis from 0 to 0.18
    # plt.ylim(bottom=1.0, top=3.0)    
    
    plt.xlabel('QBER (ε)')
    plt.ylabel('Reconciliation Efficiency (η)')
    plt.title('Reconciliation Efficiency η vs QBER (Continuous)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# --- Main Entry Point ---

if __name__ == "__main__":
    
    # 1. Run the detailed single instance log
    print("Running Detailed Log Instance...")
    cascade_single = Cascade(num_passes=10, initial_block_size=5)
    asyncio.run(run_simulation_instance(
        num_qubits=5000, 
        include_eve=True, 
        verbose=True, 
        error_correction=cascade_single
    ))
    
    # 2. Run the original plotting experiment
    asyncio.run(run_plotting_experiment())

    #3. Run the NEW Efficiency experiment
    asyncio.run(run_efficiency_experiment())