import numpy as np
import math
import asyncio
import os
import sys
import time
from typing import List, Tuple, Optional

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prng import PRNG
from cascade_refactored import ParityOracle

# --- GPU / TensorFlow / Sionna Setup ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
SIONNA_AVAILABLE = False
GPU_AVAILABLE = False
ENCODER_CLASS = None
DECODER_CLASS = None

try:
    import tensorflow as tf
    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            GPU_AVAILABLE = True
            print(f"[NR-LDPC] GPU Detected: {len(gpus)} device(s). Acceleration ENABLED.")
        except RuntimeError as e:
            print(f"[NR-LDPC] GPU Error: {e}")
    else:
        print("[NR-LDPC] No GPU detected. Running on CPU (slower).")

    # Import Sionna (v0.14+ path)
    from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
    from sionna.phy.fec.ldpc.decoding import LDPC5GDecoder
    
    ENCODER_CLASS = LDPC5GEncoder
    DECODER_CLASS = LDPC5GDecoder
    SIONNA_AVAILABLE = True
    print(f"[NR-LDPC] Sionna Library Loaded Successfully.")
    
except ImportError as e:
    print(f"[NR-LDPC] Import Error (Sionna/TF): {e}")

class NR_LDPC_ClientProtocol:
    """
    Client Protocol for 5G NR LDPC Reconciliation with HARQ-IR (Incremental Redundancy).
    
    Implements 5G Standard-compliant "Maximum Effort" decoding:
    - Uses 5G NR LDPC Mother Codes (BG1/BG2).
    - Rate Matching via Transmission of Systematic Bits (known to Bob) + Parity Bits (unknown).
    - Incremental Redundancy: 
        1. Start with high rate (few parity bits).
        2. If decoding fails, request MORE parity bits (lower rate).
        3. Combine all received parity LLRs (Chase Combining).
        4. Decode again.
        
    Note: In strict Information Reconciliation terms, this is 'Blind HARQ'.
    Bob requests parity bits until he can decode Alice's key.
    The number of bits revealed is equal to the number of parity bits requested.
    This corresponds to sending the Syndrome in a linear code,
    as Syndrome Length = Number of Parity Constraints = Number of Parity Bits in Systematic Form.
    """
    def __init__(self, verbose=True, rate=None, seed=None):
        self.verbose = verbose
        self.prng = PRNG(seed)
        self.encoder = None
        self.decoder = None
        
        # Internal State
        self.k = 0
        self.n_mother = 0     # Length of the full mother codeword
        self.G_matrix = None  # Generator Matrix [k, n_mother] 
        self.syst_indices = None # Set of indices in codeword corresponding to systematic bits
        self.out_to_in = {}   # Map: Codeword Index -> Input Key Index
        
        # Session Stats
        self.bits_revealed = 0
        self.channel_uses = 0

    def log(self, message):
        if self.verbose:
            print(f"[NR-LDPC-5G] {message}")

    def _get_theoretical_max_rate(self, qber: float) -> float:
        """Shannon Limit: 1 - H(p)"""
        if qber <= 0: return 1.0
        if qber >= 0.5: return 0.0
        h_p = -qber * math.log2(qber) - (1-qber) * math.log2(1-qber)
        return 1.0 - h_p

    def _compute_llr(self, bit, reliability=10.0):
        # LLR = log(P(1)/P(0))
        # If bit=1 -> P(1)~1 -> LLR >> 0
        # If bit=0 -> P(1)~0 -> LLR << 0
        return reliability if bit == 1 else -reliability

    async def run(self, key: List[int], qber: float, oracle: ParityOracle):
        """
        Execute Reconciliation.
        """
        if not SIONNA_AVAILABLE:
            self.log("CRITICAL: Sionna not available. Aborting.")
            return key, 0, 0, 0
            
        key_array = np.array(key, dtype=int)
        n_key = len(key_array)
        self.k = n_key
        
        if qber <= 0: qber = 0.001
        
        self.bits_revealed = 0
        self.channel_uses = 0
        
        # 1. Initialize 5G Encoder (Mother Code)
        # Industry Standard: Base Graph 1 (BG1) has a native lowest rate of 1/3 (N = 3*K).
        # We use 3.0 to align with 3GPP TS 38.212 BG1 definition.
        # Note: If QBER > 0.15, this rate (0.33) is insufficient.
        
        target_n_mother = int(self.k * 3.0) 
        
        self.log(f"Initializing 5G LDPC Engine (k={self.k}, n_mother~{target_n_mother})...")
        
        try:
            # Re-init if parameters changed
            if self.encoder is None or self.encoder.k != self.k:
                self.encoder = ENCODER_CLASS(k=self.k, n=target_n_mother)
                # Max Effort: 40 iterations (Standard is 20)
                self.decoder = DECODER_CLASS(self.encoder, num_iter=40, return_infobits=True)
                
                # Check actual n
                dummy = np.zeros((1, self.k), dtype=np.float32)
                res = self.encoder(dummy)
                self.n_mother = res.shape[1]
                self.log(f"Mother Code Configured: n={self.n_mother} (Rate {self.k/self.n_mother:.3f})")
                
                # Invalidate G Cache
                self.G_matrix = None
        except Exception as e:
            self.log(f"Setup Error: {e}")
            return key, 0, 0, 0

        # 2. Compute/Cache Generator Matrix G
        # We need this to simulate Oracle responses (mapping Parity Check to Bit Value).
        if self.G_matrix is None:
            self.log("Computing Generator Matrix G on GPU/CPU...")
            try:
                # Use large batches for efficiency
                batch_size = 256 if GPU_AVAILABLE else 32
                G_rows = []
                num_batches = math.ceil(self.k / batch_size)
                
                # We encode the Identity Matrix row by row (batched)
                # Row i of I -> Encoded -> Row i of G
                t0 = time.time()
                for i in range(num_batches):
                    # Progress
                    if i % 5 == 0:
                        sys.stdout.write(f"\rComputing G: Batch {i+1}/{num_batches}")
                        sys.stdout.flush()
                        
                        # Invalidate cache if needed
                    start = i * batch_size
                    end = min(start + batch_size, self.k)
                    size = end - start
                    
                    batch_in = np.zeros((size, self.k), dtype=np.float32)
                    for idx in range(size):
                        batch_in[idx, start + idx] = 1.0
                    
                    # Ensure encoder can handle batch
                    cw = self.encoder(batch_in).numpy()
                    G_rows.append(cw)
                
                print("") # Newline
                self.G_matrix = np.concatenate(G_rows, axis=0) # [k, n]
                self.G_matrix = (self.G_matrix > 0.5).astype(int)
                
                # Analyze Structure (Systematic vs Parity)
                # In 5G LDPC, systematic bits are usually the first K bits (mostly).
                # But Sionna puts them at specific puncturing locations sometimes?
                # For BG1, first 2*Zc columns are punctured.
                # Sionna handles this internal layout.
                # However, for our Oracle, we need to know exactly which output bit corresponds to which input bit
                # if it's systematic.
                col_weights = np.sum(self.G_matrix, axis=0)
                syst_cols = np.where(col_weights == 1)[0]
                self.syst_indices = set(syst_cols)
                
                self.out_to_in = {}
                for col in syst_cols:
                    row = np.where(self.G_matrix[:, col] == 1)[0][0]
                    self.out_to_in[col] = row
                    
                self.log(f"G Matrix Ready ({time.time()-t0:.2f}s). Systematic Bits: {len(self.syst_indices)}")
                
            except Exception as e:
                self.log(f"G Matrix Computation Failed: {e}")
                return key, 0, 0, 0

        # 3. HARQ Rate Adaptation Loop
        theory_limit = self._get_theoretical_max_rate(qber)
        
        # Start Conservative but High Rate
        # Use finer granularity for rate adaptation to avoid over-requesting
        current_rate = min(0.95, theory_limit - 0.01) 
        min_rate = 0.10
        step = 0.02 # Finer step (was 0.05) to approximate "Syndrome" efficiency
        
        # Buffer for LLRs (accumulates information)
        llr_buffer = np.zeros(self.n_mother, dtype=np.float32)
        
        # Fill Systematic Information (always available to Bob)
        # LLR magnitude based on QBER
        syst_llr_mag = math.log((1.0 - qber)/qber)
        
        for out_idx, in_idx in self.out_to_in.items():
            bit_val = key_array[in_idx]
            llr_buffer[out_idx] = self._compute_llr(bit_val, reliability=syst_llr_mag)
            
        revealed_parity_values = {} # Cache for check

        self.log(f"Starting HARQ. Max Theoretical Rate: {theory_limit:.3f}")

        # Loop: Lower Rate -> Request More Parity -> Combine -> Decode
        while current_rate >= min_rate:
            # How many bits total do we "transmit" at this rate?
            tx_len = int(self.k / current_rate)
            if tx_len > self.n_mother: tx_len = self.n_mother
            
            self.log(f"HARQ Round: Rate {current_rate:.2f} (Block {tx_len})...")
            
            # Identify parity bits in the new transmission window [0...tx_len]
            # that we haven't requested yet.
            needed = []
            
            # Optimization: Only iterate from previous tx_len to current tx_len?
            # But tx_len is derived from current_rate, which changes.
            # And previous loop might have revealed some.
            # But logic seems safe: if j not in revealed_parity_values.
            
            # CRITICAL FIX: Smart selection of parity bits
            # Instead of taking the first available bits [0...tx_len],
            # we should prioritize bits that provide new information.
            # In 5G LDPC BG1, the first 2*Z columns are punctured.
            # However, since we use Sionna's encoder output directly, G includes these columns.
            # Requesting them is fine, they are just bits.
            # But the order matters?
            # We just take linear scan.
            
            # Rate limiting check (don't overshoot theory too much)
            # If current_rate < optimal_rate, we might be wasting bits.
            # But we are in a loop decreasing rate because previous decoding failed.
            
            # The only way to reduce bits revealed is to succeed earlier (higher rate).
            # This requires better decoding (more iterations?) or better code structure.
            # We already increased iterations to 40.
            # Let's try 100 for "Maximum Effort".
            
            # OR logic issue: Are we counting systematic bits in "bits revealed"?
            # No. `self.bits_revealed += len(values)`. `values` are parity only.
            
            # Is it possible `tx_len` calculation is too aggressive?
            # tx_len = k / rate.
            # parity_count = k/rate - k.
            # If rate = 0.5, parity = k. Total = 2k.
            # That is correct for Rate 1/2 code.
            
            # Let's ensure we don't request systematic bits again if they are not in syst_indices?
            # syst_indices covers all weight-1 columns.
            # What if some systematic bits are not weight-1 in G? (Unlikely for systematic code).
            
            effective_max = min(tx_len, self.n_mother)
            
            for j in range(effective_max):
                if j in self.syst_indices:
                    continue # Already known (systematic)
                    
                if j in revealed_parity_values:
                    continue # Already revealed
                    
                needed.append(j)
            
            if needed:
                # Construct Oracle Requests
                # Parity Bit j is sum(Key[i] for i where G[i,j]=1)
                blocks = []
                for p_idx in needed:
                    participants = np.where(self.G_matrix[:, p_idx] == 1)[0].tolist()
                    blocks.append(participants)
                
                self.log(f"Requesting {len(blocks)} parity bits (Incremental).")
                self.channel_uses += 1
                
                try:
                    # Async Request
                    values = await oracle.get_parities(blocks)
                    self.bits_revealed += len(values)
                    
                    # Update LLRs
                    for p_idx, val in zip(needed, values):
                        revealed_parity_values[p_idx] = val
                        # Parity from Alice is ERROR-FREE (Infinite reliability)
                        llr_buffer[p_idx] = self._compute_llr(val, reliability=100.0)
                        
                except Exception as e:
                    self.log(f"Oracle Error: {e}")
                    break
            else:
                self.log("No new parity needed for this rate step.")

            # DECODE
            try:
                dc_in = tf.convert_to_tensor([llr_buffer], dtype=tf.float32)
                decoded_est = self.decoder(dc_in).numpy()[0].astype(int)
                
                # Check Syndrome (CRC)
                # Using all revealed parity bits as a hash check
                if revealed_parity_values:
                    # Re-encode estimation
                    est_codeword = np.dot(decoded_est, self.G_matrix) % 2
                    
                    errors = 0
                    # Optim: only check RECENTLY revealed bits + random subset to improve speed?
                    # No, check all to be secure.
                    for p_idx, val in revealed_parity_values.items():
                        if est_codeword[p_idx] != val:
                            errors += 1
                    
                    if errors == 0:
                        self.log("HARQ Success! All parity checks passed.")
                        return decoded_est.tolist(), self.bits_revealed, 0, self.channel_uses
                    else:
                        # If we have revealed ALL possible parity bits (n_mother limit reached)
                        # and still fail, we should probably abort to save time,
                        # but we might as well return best effort or fail.
                        if len(revealed_parity_values) + len(self.syst_indices) >= self.n_mother:
                            self.log("HARQ Fail: Maximum parity reached. Aborting.")
                            break
                        
                    # Usually at high rate we at least verify CRC if available.
                    # But here parity serves as CRC. If none requested, we might have issues.
                    # Let's force verify at least one parity block or loop again?
                    # Or just return.
                    self.log("No parity to verify. Assuming correct? (Dangerous). Lowering rate.")
                    # Force drop rate to be safe, or return if confident?
                    # If we return here, we might return errors.
                    # Better to fall through and get parity.
                     
            except Exception as e:
                self.log(f"Decoding Error: {e}")

            # Prepare for next round
            current_rate -= step

        self.log("HARQ Failed after minimum rate.")
        return key, self.bits_revealed, 0, self.channel_uses

if __name__ == "__main__":
    async def main():
        print("--- 5G NR LDPC Unit Test ---")
        N = 2000
        prng = PRNG(123)
        alice = [prng.randint(0,1) for _ in range(N)]
        qber = 0.20
        bob = list(alice)
        errs = 0
        for i in range(N):
            if prng.random() < qber:
                bob[i] = 1 - bob[i]
                errs += 1
        
        print(f"Details: N={N}, QBER={qber}, Errors={errs}")

        # Mock Oracle
        class MockOracle:
            def __init__(self, k): self.k = np.array(k)
            async def get_parities(self, blocks):
                return [int(np.sum(self.key[b])%2) for b in blocks]
        
        oracle = MockOracle(alice)
        oracle.key = np.array(alice)
        
        prot = NR_LDPC_ClientProtocol(verbose=True)
        res, bits, _, _ = await prot.run(bob, qber, oracle)
        
        final_errs = np.sum(np.abs(np.array(res) - np.array(alice)))
        print(f"Final Errors: {final_errs}")
        print(f"Bits Revealed: {bits}")

    if SIONNA_AVAILABLE:
        asyncio.run(main())
