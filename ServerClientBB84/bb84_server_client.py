
import asyncio
import logging
import datetime
import math
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Any
from abc import ABC, abstractmethod
from typing import Protocol

# Import the new Cascade Client
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from prng import PRNG, Basis
from cascade_refactored import CascadeClientProtocol, ParityOracle
from qc_ldpc import QCLDPCClientProtocol
from qc_ldpc_v2 import QCLDPCv2ClientProtocol
import time

# --- Logging Configuration ---
logging.basicConfig(level=logging.ERROR, format='%(message)s')

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

# --- Actor Framework (Unchanged) ---
class Actor:
    def __init__(self, name, seed=None):
        self.name = name
        self.mailbox = asyncio.Queue()
        self._running = False
        self.prng = PRNG(seed)

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

# --- Physical Actors (Unchanged) ---
class QuantumChannel(Actor):
    def __init__(self, name, length_km, attenuation_db, optical_error_rate, next_actor, seed=None):
        super().__init__(name, seed)
        self.length = length_km
        self.att_db = attenuation_db
        self.optical_error_rate = optical_error_rate
        self.next_actor = next_actor
        self.transmittance = 10**(-(length_km * attenuation_db)/10)

    async def handle_message(self, msg):
        if not isinstance(msg, Photon):
            await self.send(self.next_actor, msg)
            return

        if self.prng.random() > self.transmittance: return # Photon lost

        final_bit = msg.bit
        if self.prng.random() < self.optical_error_rate:
            final_bit = 1 - final_bit 

        out_photon = Photon(msg.id, final_bit, msg.basis, msg.creation_time)
        await self.send(self.next_actor, out_photon)

class Detector(Actor):
    def __init__(self, name, efficiency, dark_count_prob, parent_bob, seed=None):
        super().__init__(name, seed)
        self.eta = efficiency
        self.p_dc = dark_count_prob
        self.parent = parent_bob

    async def handle_message(self, msg):
        current_time = datetime.datetime.now().strftime("%H:%M:%S.%f")
        click = False
        measured_bit = -1
        bob_basis = None
        source = "None"

        if self.prng.random() < self.p_dc:
            click = True
            source = "DarkCount"
            measured_bit = self.prng.get_bit()
            bob_basis = self.prng.get_basis()

        if not click and isinstance(msg, Photon):
            if self.prng.random() < self.eta:
                click = True
                source = "Signal"
                bob_basis = self.prng.get_basis()
                if bob_basis == msg.basis:
                    measured_bit = msg.bit
                else:
                    measured_bit = self.prng.get_bit()
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
    def __init__(self, name, next_actor, intercept_rate=0.2, seed=None):
        super().__init__(name, seed)
        self.next_actor = next_actor
        self.intercept_rate = intercept_rate  

    async def handle_message(self, msg):
        if not isinstance(msg, Photon):
            await self.send(self.next_actor, msg)
            return
        
        if self.prng.random() > self.intercept_rate:
            await self.send(self.next_actor, msg)
            return
        
        eve_basis = self.prng.get_basis()
        measured_bit = msg.bit
        
        if eve_basis != msg.basis:
            measured_bit = self.prng.get_bit()
            
        new_photon = Photon(msg.id, measured_bit, eve_basis, msg.creation_time)
        await self.send(self.next_actor, new_photon)

# --- ALICE (SERVER) ---

class AliceServer(Actor):
    def __init__(self, name, channel, num_qubits, verbose=False, seed=None):
        super().__init__(name, seed)
        self.channel = channel
        self.num_qubits = num_qubits
        self.sent_log = {}  # photon_id -> Photon
        self.sifted_key = [] # Stores (bit, photon_id) or just bits. 
                             # Need to keep original indices or photon ids mapping? 
                             # The PRD says "Alice -> Bob: Matching basis indices"
        self.sifted_bits = [] # Actual bits
        self.verbose = verbose

    async def run_quantum_transmission(self):
        if self.verbose: 
            print(f"\n[ALICE] Generating and sending {self.num_qubits} qubits...")
            print(f"[ALICE] Format: Bit(Basis) - e.g., 1(+) or 0(x)")
        
        sample_log = []
        for i in range(self.num_qubits):
            bit = self.prng.get_bit()
            basis = self.prng.get_basis()
            t_now = datetime.datetime.now().strftime("%H:%M:%S.%f")
            p = Photon(id=i, bit=bit, basis=basis, creation_time=t_now)
            
            # Store only what is needed
            self.sent_log[i] = p
            
            if i < 15:
                sample_log.append(f"{bit}({basis.value})")
                
            await self.send(self.channel, p)
            if i % 1000 == 0: await asyncio.sleep(0.0001)
            
        if self.verbose: 
            print(f"[ALICE] First 15 Qubits Sent: {', '.join(sample_log)} ...")
            print(f"[ALICE] Transmission complete.")

    # --- API ENDPOINTS (Simulated) ---
    
    async def handle_api_request(self, endpoint, data):
        """
        Simulates an HTTP server router.
        """
        if endpoint == "/sift":
            return self._api_sift(data)
        elif endpoint == "/sample":
            return self._api_sample(data)
        elif endpoint == "/parity":
            return self._api_parity(data)
        elif endpoint == "/block-parity":
            return self._api_block_parity(data)
        else:
            raise ValueError(f"Unknown endpoint {endpoint}")

    def _api_sift(self, data):
        # Bob sends: { "bases": { photon_id: "RECTILINEAR"|"DIAGONAL", ... } }
        # Actually Bob should just send list of (photon_id, basis) to be efficient
        bob_bases = data["bases"] # List of (id, basis_str)
        
        if self.verbose:
            print(f"[ALICE] Received sifting request with {len(bob_bases)} bases from Bob.")
        
        matching_indices = []
        self.sifted_bits = [] 
        # Alice reconstructs her sifted key based on the order Bob presents
        # The indices Bob uses in the future (0, 1, 2...) correspond to the list of matches
        
        for pid, b_basis_str in bob_bases:
            if pid in self.sent_log:
                tx_photon = self.sent_log[pid]
                # Convert string back to Enum if needed, or just compare value
                if tx_photon.basis.value == b_basis_str:
                    matching_indices.append(pid)
                    self.sifted_bits.append(tx_photon.bit)
        
        if self.verbose:
            print(f"[ALICE] Bases matched: {len(matching_indices)}. Discarded: {len(bob_bases) - len(matching_indices)}.")
            if len(self.sifted_bits) > 0:
                print(f"[ALICE] Sifted Key Preview: {self.sifted_bits[:20]}...")
        
        # We need to persist the sifted key for subsequent steps
        # self.sifted_bits is now the reference key
        
        return {"matching_indices": matching_indices}

    def _api_sample(self, data):
        # Bob requests specific bits to be revealed for QBER estimation
        indices = data["indices"]
        
        if self.verbose:
            print(f"[ALICE] Received QBER sample request for {len(indices)} indices.")
            
        revealed_bits = []
        for idx in indices:
            if 0 <= idx < len(self.sifted_bits):
                revealed_bits.append(self.sifted_bits[idx])
            else:
                revealed_bits.append(None) # Error?
        
        # IMPORTANT: Bits revealed during parameter estimation must be DISCARDED
        # The generic implementation `calculate_metrics` does logical discarding.
        # But Alice needs to know which ones to discard for step 2 (Cascade)?
        # The PRD says: "Revealed bits are discarded."
        # So we should probably remove them or mark them.
        # However, for simplicity and index coherence, we often keep indices stable and just ignore them.
        # In this implementation, we will assume Bob drives the "clean" key creation locally 
        # but for Cascade, Bob calculates parity on the "Remaining" key.
        # Alice needs to know which bits form the "Remaining" key.
        # OR: Cascade uses the *sifted* key but skips the revealed ones?
        # Typically Cascade runs on the Sifted Key after removing the sampled bits.
        # If Bob removes them, he has a smaller key. Alice needs to match that.
        
        # Let's assume the API for sample implies discarding.
        # But wait, if Alice modifies `self.sifted_bits` she loses index sync if Bob doesn't do exactly same.
        # Better: Bob sends "indices to reveal". Both agree to drop them.
        # But the API is stateless-ish. 
        # Let's add a `discard_indices` to this call or a separate call? 
        # PRD says: "Revealed bits are discarded."
        
        # Implementation Detail: 
        # After sampling, we usually assume the "Key" used for Cascade is the "Sifted Sub Samped Key".
        # Alice needs to update her state to this new key for parity checks to work on the correct indices.
        
        # Let's perform the drop here.
        indices_set = set(indices)
        new_key = [b for i, b in enumerate(self.sifted_bits) if i not in indices_set]
        self.sifted_bits = new_key
        
        if self.verbose:
             print(f"[ALICE] Revealed bits: {revealed_bits[:15]}...")
             print(f"[ALICE] Removing revealed bits. Working key size: {len(new_key)}.")
        
        return {"bits": revealed_bits}

    def _par_calc(self, indices: List[int]) -> int:
        p_val = 0
        N = len(self.sifted_bits)
        key_int = 0
        # Convert bit list to int? Or just loop. 
        # Cascade uses integer math for speed, but here we can just loop.
        # But wait, cascade parity logic in the client was carefully matching bits.
        # The client code does: bit = (Z_int >> (N - 1 - idx))
        # This implies index 0 is MSB.
        # So self.sifted_bits[0] corresponds to MSB.
        
        for idx in indices:
            if 0 <= idx < N:
                p_val ^= self.sifted_bits[idx]
        return p_val

    def _api_parity(self, data):
        indices = data["indices"]
        return {"parity": self._par_calc(indices)}

    def _api_block_parity(self, data):
        blocks = data["blocks"]
        results = [self._par_calc(b) for b in blocks]
        return {"parities": results}

# --- API CLIENT ADAPTER ---

class APIClient(ParityOracle):
    def __init__(self, server: AliceServer):
        self.server = server
        
    async def post(self, endpoint, data):
        # Simulate Network Delay
        # await asyncio.sleep(0.001) 
        return await self.server.handle_api_request(endpoint, data)

    # Implement ParityOracle Protocol
    async def get_parity(self, indices: List[int]) -> int:
        resp = await self.post("/parity", {"indices": indices})
        return resp["parity"]

    async def get_parities(self, blocks: List[List[int]]) -> List[int]:
        resp = await self.post("/block-parity", {"blocks": blocks})
        return resp["parities"]

# --- BOB (CLIENT) ---

class BobClient(Actor):
    def __init__(self, name, api_client: APIClient, protocol="cascade", seed=None, verbose=False):
        super().__init__(name, seed)
        self.api = api_client
        self.protocol = protocol.lower()
        self.verbose = verbose
        self.received_events = [] # List[DetectionEvent]

    async def handle_message(self, msg):
        if isinstance(msg, DetectionEvent):
            self.received_events.append(msg)

    async def run_classical_post_processing(self, num_input_qubits):
        # 1. Sifting
        # Prepare payload: list of (id, basis)
        bases_payload = []
        print(f"\n[BOB] Processing {len(self.received_events)} detection events.")
        
        valid_evs = [e for e in self.received_events if e.bit != -1]
        sample_valid = [f"{e.bit}({e.basis.value})" for e in valid_evs[:15]]
        print(f"[BOB] First 15 Valid Detections: {', '.join(sample_valid)} ...")
        
        for ev in self.received_events:
            if ev.bit != -1: # Valid detection
                bases_payload.append((ev.photon_id, ev.basis.value))
        
        print(f"[BOB] Sending {len(bases_payload)} bases to Alice for sifting...")
        response = await self.api.post("/sift", {"bases": bases_payload})
        matching_ids = set(response["matching_indices"])
        print(f"[BOB] Sifting complete. Bases matched on {len(matching_ids)} events.")
        
        # Filter local key
        sifted_key_bob = []
        for ev in self.received_events:
            if ev.photon_id in matching_ids:
                sifted_key_bob.append(ev.bit)
        
        print(f"[BOB] Sifted Key Preview: {sifted_key_bob[:20]}...")
        
        # 2. Parameter Estimation (QBER)
        n_sifted = len(sifted_key_bob)
        sample_size = 0.2
        k = int(n_sifted * sample_size)
        
        # Enforce minimum sample size of 10 bits
        if n_sifted < 10:
             print(f"[BOB] Error: Sifted key too short ({n_sifted}) for QBER estimation.")
             return {
                "sifted_length": n_sifted,
                "qber": 1.0, # Fail safe
                "final_length": 0,
                "revealed": 0,
                "corrected": 0,
                "channel_uses": 0,
                "corrected_key": [],
                "exec_time": 0
            }
        
        if k < 10:
            k = 10
            print(f"[BOB] Adjusted sample size to minimum 10 bits.")
        
        print(f"[BOB] Sampling {k} bits for QBER estimation...")
        
        if k > 0:
            indices_to_sample = self.prng.sample(range(n_sifted), k)
            resp = await self.api.post("/sample", {"indices": indices_to_sample})
            alice_bits = resp["bits"]
            
            # Calculate QBER
            errors = 0
            for i, idx in enumerate(indices_to_sample):
                if sifted_key_bob[idx] != alice_bits[i]:
                    errors += 1
            qber = errors / k
            print(f"[BOB] QBER Analysis: {errors} errors in {k} samples. Estimated QBER = {qber:.2%}")
            
            # Remove sampled bits from key
            indices_set = set(indices_to_sample)
            clean_key_bob = [b for i, b in enumerate(sifted_key_bob) if i not in indices_set]
            print(f"[BOB] Discarding revealed bits. Working Key Size: {len(clean_key_bob)}")
        else:
            qber = 0.0
            clean_key_bob = list(sifted_key_bob)
            print("[BOB] Not enough bits to sample. QBER set to 0.0")

        # 3. Reconciliation
        print(f"[BOB] Initializing {self.protocol.title()} Protocol...")
        t_start = time.time()
        
        est_qber = qber if qber > 0 else 0.01

        if self.protocol == "cascade":
            protocol = CascadeClientProtocol(num_passes=4, verbose=self.verbose)
            corrected_key, revealed, errors_cor, uses = await protocol.run(clean_key_bob, est_qber, self.api)
        elif self.protocol == "ldpc":
            # Generate deterministic seed for protocol from Bob's PRNG
            proto_seed = self.prng.randint(0, 2**32 - 1)
            protocol = QCLDPCClientProtocol(verbose=self.verbose, rate="adaptive", seed=proto_seed)
            corrected_key, revealed, errors_cor, uses = await protocol.run(clean_key_bob, est_qber, self.api)
        elif self.protocol == "ldpcv2":
            # Generate deterministic seed for protocol from Bob's PRNG
            proto_seed = self.prng.randint(0, 2**32 - 1)
            # Use "adaptive" rate
            protocol = QCLDPCv2ClientProtocol(verbose=self.verbose, rate="adaptive", seed=proto_seed)
            corrected_key, revealed, errors_cor, uses = await protocol.run(clean_key_bob, est_qber, self.api)
        else:
             print(f"Unknown protocol {self.protocol}, defaulting to Cascade")
             protocol = CascadeClientProtocol(num_passes=4, verbose=self.verbose)
             corrected_key, revealed, errors_cor, uses = await protocol.run(clean_key_bob, est_qber, self.api)

        exec_time = time.time() - t_start

        # Return metrics
        return {
            "sifted_length": n_sifted,
            "qber": qber,
            "final_length": len(corrected_key),
            "revealed": revealed,
            "corrected": errors_cor,
            "channel_uses": uses,
            "corrected_key": corrected_key,
            "exec_time": exec_time
        }

# --- MAIN ORCHESTRATION ---

async def run_server_client_simulation():
    print("Initializing Server-Client Simulation...")
    
    # Seeds
    seed_alice = 1001
    seed_bob = 1002
    seed_channel = 1003
    seed_detector = 1004
    
    # Setup Channel & Physical Layer
    bob_placeholder = Actor("BobPlaceholder") # Bob will attach later or we pass msg manually
    
    # Actually, we need the Actor wiring to work.
    # BobClient is an Actor.
    
    # Wire it up: Alice -> Channel -> Detector -> BobClient
    
    channel = QuantumChannel("Fiber", 50, 0.2, 0.01, next_actor=None, seed=seed_channel)
    # Detector needs parent (Bob)
    # We create Bob first but he needs API client which needs Alice
    # Alice needs Channel.
    
    # 1. Create Channel (partial)
    # 2. Create Alice (needs channel)
    # 3. Create API Client (needs Alice)
    # 4. Create Bob (needs API client)
    # 5. Create Detector (needs Bob)
    # 6. Link Channel -> Detector
    
    # But QuantumChannel ctor takes next_actor immediately.
    # We can set it after.
    
    channel = QuantumChannel("Fiber", 50, 0.2, 0.01, next_actor=None, seed=seed_channel)
    alice = AliceServer("AliceServer", channel, num_qubits=5000, verbose=True, seed=seed_alice)
    api = APIClient(alice)
    bob = BobClient("BobClient", api, seed=seed_bob)
    detector = Detector("Detector", 0.8, 0.01, parent_bob=bob, seed=seed_detector)
    
    channel.next_actor = detector # Closing the loop
    
    # Start Actors
    actors = [alice, channel, detector, bob]
    tasks = [asyncio.create_task(a.start()) for a in actors]
    
    # Run Protocol Phase 1: Transmission
    await alice.run_quantum_transmission()
    
    # Wait for photons to arrive
    await asyncio.sleep(1.0)
    
    # Setup Classical Phase
    # Bob initiates
    results = await bob.run_classical_post_processing(alice.num_qubits)
    
    print("\n--- RESULTS ---")
    print(f"Sifted Key Length: {results['sifted_length']}")
    print(f"QBER (Est): {results['qber']:.4%}")
    print(f"Final Key Length: {results['final_length']}")
    print(f"Cascade Revealed: {results['revealed']} bits")
    print(f"Cascade Corrected: {results['corrected']} errors")
    print(f"Channel Uses: {results['channel_uses']}")
    
    # Verification (God Mode)
    # Alice's final key is self.sifted_bits (she updated it during sampling)
    alice_final_key = alice.sifted_bits
    bob_final_key = results["corrected_key"]
    
    matches = 0
    for a, b in zip(alice_final_key, bob_final_key):
        if a == b: matches += 1
    
    print(f"Final Key Match: {matches}/{len(alice_final_key)} ({(matches/len(alice_final_key) if len(alice_final_key)>0 else 0):.2%})")
    
    # Stop actors
    for a in actors: await a.send(a, ("STOP",))
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(run_server_client_simulation())
