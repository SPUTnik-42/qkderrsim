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

class NR_LDPC_Standard_ClientProtocol:
    """
    Standard Client Protocol for 5G NR LDPC Reconciliation.
    
    Implements 5G Standard-compliant Fixed Rate decoding:
    - Uses 5G NR LDPC Mother Codes (BG1).
    - Rate Matching via Transmission of Systematic Bits (known to Bob) + Parity Bits (unknown).
    - Supports setting specific target rates (default 0.33 for true 5G BG1 rate 1/3).
    """
    def __init__(self, verbose=True, rate=0.333, seed=None):
        """
        Initialize the protocol.
        
        Args:
            verbose (bool): Whether to print logs.
            rate (float): The target code rate (R = k/n). Lower rate = more parity = better correction.
                          Standard 5G BG1 base rate is roughly 1/3 (0.33).
            seed (int): Seed for PRNG.
        """
        self.verbose = verbose
        self.target_rate = rate if rate is not None else 0.333
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
            print(f"[NR-LDPC-STD] {message}")

    def _compute_llr(self, bit, reliability=10.0):
        # LLR = log(P(1)/P(0))
        # If bit=1 -> P(1)~1 -> LLR >> 0
        # If bit=0 -> P(1)~0 -> LLR << 0
        return reliability if bit == 1 else -reliability

    def _calculate_puncturing_limit(self, qber: float) -> int:
        """
        Calculate how many parity bits are needed based on QBER.
        Using Shannon Limit as a baseline but strictly sticking to standard rates.
        
        R <= 1 - H(QBER)
        n_needed = k / R
        
        However, we want to be conservative to ensure decoding success.
        We will use a small margin but NOT an arbitrary efficiency factor.
        We just interpret the QBER to find the minimum necessary block size.
        """
        if qber <= 0: return 0
        h_p = -qber * math.log2(qber) - (1-qber) * math.log2(1-qber)
        
        # Theoretical max rate
        max_rate = 1.0 - h_p
        
        # Avoid division by zero close to 1
        if max_rate < 0.05: max_rate = 0.05
        
        # Calculate N needed to satisfy this rate
        n_needed = int(self.k / max_rate)
        
        # Limit to the mother code length available
        if n_needed > self.n_mother:
            n_needed = self.n_mother
            
        # Parity bits needed = n_needed - k_systematic (approx)
        # Actually strictly: n_needed - number_of_systematic_transmitted
        # Since we transmit all systematic bits (key), it is n_needed - k
        
        parity_needed = max(0, n_needed - self.k)
        
        # Ensure we don't request more than available in the mother code
        # The mother code has (n_mother - k) parity bits roughly.
        
        # Available parity indices
        available_parity = self.n_mother - len(self.syst_indices)
        if parity_needed > available_parity:
            parity_needed = available_parity
            
        return parity_needed

    async def run(self, key: List[int], qber: float, oracle: ParityOracle):
        """
        Execute Reconciliation in one shot with Rate Matching (Puncturing).
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
        
        # 1. Initialize 5G Encoder (Mother Code) - ALWAYS USE STANDARD LOW RATE (High Reliability)
        # We initialize the encoder with the default base rate (e.g. 1/3) to have max parity available.
        # Then we puncture (don't send) based on QBER.
        # If user specified a specific hard-coded rate in __init__, we respect it as the "Mother Code" limit,
        # but we still puncture if QBER allows.
        
        # Standard BG1 is Rate 1/3 => n = 3k
        # Use the configured target_rate to set the MOTHER code dimension.
        mother_n = int(self.k / self.target_rate)
        
        self.log(f"Initializing 5G LDPC Engine (k={self.k}, Mother n={mother_n}, Base Rate={self.target_rate:.3f})...")
        
        try:
            # Re-init if parameters changed
            if self.encoder is None or self.encoder.k != self.k or self.encoder._n != mother_n:
                self.encoder = ENCODER_CLASS(k=self.k, n=mother_n)
                # Standard decoding iterations
                self.decoder = DECODER_CLASS(self.encoder, num_iter=20, return_infobits=True)
                
                # Check actual n
                dummy = np.zeros((1, self.k), dtype=np.float32)
                res = self.encoder(dummy)
                self.n_mother = res.shape[1]
                self.log(f"Code Configured: n={self.n_mother} (Actual Rate {self.k/self.n_mother:.3f})")
                
                # Invalidate G Cache
                self.G_matrix = None
        except Exception as e:
            self.log(f"Setup Error: {e}")
            return key, 0, 0, 0

        # 2. Compute/Cache Generator Matrix G
        # We need this to simulate Oracle responses (mapping Parity Check to Bit Value).
        if self.G_matrix is None:
            self.log("Computing Generator Matrix G...")
            try:
                # Use batches for simplicity
                batch_size = 256 # Standard batch size
                G_rows = []
                num_batches = math.ceil(self.k / batch_size)
                
                # We encode the Identity Matrix row by row (batched)
                # Row i of I -> Encoded -> Row i of G
                for i in range(num_batches):
                    start = i * batch_size
                    end = min(start + batch_size, self.k)
                    size = end - start
                    
                    batch_in = np.zeros((size, self.k), dtype=np.float32)
                    for idx in range(size):
                        batch_in[idx, start + idx] = 1.0
                    
                    # Ensure encoder can handle batch
                    cw = self.encoder(batch_in).numpy()
                    G_rows.append(cw)
                
                self.G_matrix = np.concatenate(G_rows, axis=0) # [k, n]
                self.G_matrix = (self.G_matrix > 0.5).astype(int)
                
                # Analyze Structure (Systematic vs Parity)
                col_weights = np.sum(self.G_matrix, axis=0)
                syst_cols = np.where(col_weights == 1)[0]
                self.syst_indices = set(syst_cols)
                
                self.out_to_in = {}
                for col in syst_cols:
                    row = np.where(self.G_matrix[:, col] == 1)[0][0]
                    self.out_to_in[col] = row
                    
                self.log(f"G Matrix Ready. Systematic Bits: {len(self.syst_indices)}")
                
            except Exception as e:
                self.log(f"G Matrix Computation Failed: {e}")
                return key, 0, 0, 0

        # 3. One-Shot Punctured Exchange
        
        # Buffer for LLRs
        # Initialize with 0s. 0 means "No Information" (Punctured).
        llr_buffer = np.zeros(self.n_mother, dtype=np.float32)
        
        # Fill Systematic Information (known to Bob, with noise)
        syst_llr_mag = math.log((1.0 - qber)/qber)
        
        for out_idx, in_idx in self.out_to_in.items():
            bit_val = key_array[in_idx]
            llr_buffer[out_idx] = self._compute_llr(bit_val, reliability=syst_llr_mag)
            
        # Identify available parity bits
        available_parity_indices = []
        for j in range(self.n_mother):
            if j not in self.syst_indices:
                available_parity_indices.append(j)
        
        # --- RATE MATCHING / PUNCTURING LOGIC ---
        # Calculate how many parity bits we actually need based on QBER
        # We start filling the circular buffer (linear in available_parity_indices)
        # up to the calculated limit.
        
        num_parity_to_send = self._calculate_puncturing_limit(qber)
        
        self.log(f"Rate Matching: QBER={qber:.4f} => Need ~{num_parity_to_send} parity bits (out of {len(available_parity_indices)} available).")
        
        # Select the bits (Standard: Linear selection from start of circular buffer)
        # In 5G LDPC, the first 2*Zc columns are punctured, but Sionna handles the output map.
        # We just take the first N bits from the encoder output that are parity.
        
        parity_indices_to_request = available_parity_indices[:num_parity_to_send]
        
        # Request Selected Parity Bits
        if parity_indices_to_request:
            blocks = []
            for p_idx in parity_indices_to_request:
                participants = np.where(self.G_matrix[:, p_idx] == 1)[0].tolist()
                blocks.append(participants)
            
            self.log(f"Requesting {len(blocks)} parity bits (Punctured Transmission).")
            self.channel_uses += 1
            
            try:
                # Async Request
                values = await oracle.get_parities(blocks)
                self.bits_revealed += len(values)
                
                # Update LLRs for received bits
                for p_idx, val in zip(parity_indices_to_request, values):
                    llr_buffer[p_idx] = self._compute_llr(val, reliability=100.0)
                    
            except Exception as e:
                self.log(f"Oracle Error: {e}")
                return key, 0, 0, 0
        else:
            self.log("No parity needed (QBER extremely low).")

        # 4. Decode
        try:
            dc_in = tf.convert_to_tensor([llr_buffer], dtype=tf.float32)
            decoded_est = self.decoder(dc_in).numpy()[0].astype(int)
            
            # Check correctness (Verify parity constraints)
            est_codeword = np.dot(decoded_est, self.G_matrix) % 2
            
            errors = 0
            # Verify against requested parities
            # Note: We can only verify against bits we actually received!
            if parity_indices_to_request:
                for p_idx, val in zip(parity_indices_to_request, values):
                    if est_codeword[p_idx] != val:
                        errors += 1
            
            # Check against systematic (if we had access to real key, but here we don't know real key)
            # This 'errors' is just Syndrome Check on the received vector.
            # If rate is high (heavy puncturing), we might pass syndrome but fail decoding if not enough info.
            # But the Parity Oracle returns true parity.
            
            if errors == 0:
                self.log("Decoding Success! Parity checks passed.")
            else:
                self.log(f"Decoding Failed: {errors} parity mismatches.")
                
            return decoded_est.tolist(), self.bits_revealed, errors, self.channel_uses
                    
        except Exception as e:
            self.log(f"Decoding Error: {e}")
            return key, self.bits_revealed, 0, self.channel_uses

if __name__ == "__main__":
    async def main():
        print("--- 5G NR LDPC Standard Test ---")
        N = 2000
        prng = PRNG(123)
        alice = [prng.randint(0,1) for _ in range(N)]
        qber = 0.05
        bob = list(alice)
        errs = 0
        for i in range(N):
            if prng.random() < qber:
                bob[i] = 1 - bob[i]
                errs += 1
        
        print(f"Details: N={N}, QBER={qber}, Errors={errs}")

        class MockOracle:
            def __init__(self, k): self.k = np.array(k)
            async def get_parities(self, blocks):
                return [int(np.sum(self.key[b])%2) for b in blocks]
        
        oracle = MockOracle(alice)
        oracle.key = np.array(alice)
        
        # Test 
        prot = NR_LDPC_Standard_ClientProtocol(verbose=True, rate=0.33)
        res, bits, fail, _ = await prot.run(bob, qber, oracle)
        
        final_errs = np.sum(np.abs(np.array(res) - np.array(alice)))
        print(f"Final Errors: {final_errs}")
        print(f"Bits Revealed: {bits}")

    if SIONNA_AVAILABLE:
        asyncio.run(main())