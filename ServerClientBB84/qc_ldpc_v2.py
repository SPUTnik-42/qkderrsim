import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix, dok_matrix
import ldpc
from prng import PRNG
from typing import List, Tuple, Optional
from cascade_refactored import ParityOracle

import math

class QCLDPCv2ClientProtocol:
    """
    Implements QC-LDPC error correction using the 'ldpc' (LDPCv2) library.
    Supports adaptive rate adaptation based on estimated QBER.
    """
    def __init__(self, verbose=True, rate=0.5, seed=None):
        self.verbose = verbose
        self.prng = PRNG(seed)
        # If rate is None or "adaptive", we calculate on the fly
        self.initial_rate = rate 
        self.H: Optional[csr_matrix] = None
        self.N: int = 0
        self.M: int = 0
        
        # Metrics
        self.bits_revealed = 0
        self.total_errors_corrected = 0
        self.channel_uses = 0

    def log(self, message):
        if self.verbose:
            print(f"[QC-LDPC-v2] {message}")

    def _calculate_adaptive_rate(self, qber: float) -> float:
        """
        Calculates optimal coding rate K/N based on QBER.
        R = 1 - (efficiency * H(qber))
        We want Code Rate.
        Syndrome Rate (M/N) = efficiency * H(qber).
        We use an efficiency factor of 1.2 (20% overhead) which is achievable with modern LDPC codes.
        """
        if qber <= 0: return 0.9 # High rate if no errors
        if qber >= 0.5: return 0.0 # Cannot correct
        
        # Binary Entropy
        h_p = -qber * math.log2(qber) - (1-qber) * math.log2(1-qber)
        
        inefficiency = 1.20 # Improved from 1.50 (Safety margin)
        syndrome_rate = h_p * inefficiency
        
        # Robustness: If H(p) is very high, we approach the Shannon limit.
        # If the required syndrome rate is close to 1.0, we should just reveal most/all bits
        # to ensure decoding success, rather than capping at 0.95 and failing (which looks like efficiency < 1).
        if syndrome_rate >= 0.95: 
            syndrome_rate = 1.0 # Fallback: Reveal everything (Efficiency > 1 guaranteed)
        elif syndrome_rate < 0.05: 
            syndrome_rate = 0.05
        
        code_rate = 1.0 - syndrome_rate
        return code_rate

    def construct_H_matrix(self, n_key: int, qber: float) -> Tuple[csr_matrix, int]:
        """
        Constructs a QC-LDPC Parity Check Matrix H (Sparse & Irregular).
        Returns sparse H and the adjusted valid key length (N).
        """
        # Determine Rate
        if self.initial_rate == "adaptive" or self.initial_rate is None or self.initial_rate == "auto":
            target_code_rate = self._calculate_adaptive_rate(qber)
            self.log(f"Adaptive Rate Calculation: QBER={qber:.4f} -> Code Rate={target_code_rate:.3f}")
        else:
            target_code_rate = float(self.initial_rate)

        # QC-LDPC Parameters
        K_base = 24 # Number of block columns
        
        # J_base is number of block rows.
        # R = (K - J) / K = 1 - J/K => J/K = 1 - R => J = K(1-R)
        # Use ceil to ensure we meet the syndrome rate requirement (avoid rounding down)
        J_base = math.ceil(K_base * (1.0 - target_code_rate)) 
        
        if J_base < 3: J_base = 3 
        # Check against K_base
        if J_base >= K_base: J_base = K_base - 1

        # Lifting size 'b'
        b = n_key // K_base
        if b < 1: b = 1

        
        N = K_base * b
        M = J_base * b
        
        self.log(f"Constructing Irregular Sparse H (b={b}, Base={J_base}x{K_base}, Final={M}x{N}, Rate={(N-M)/N:.2f})")

        # Base Matrix Construction (Irregular with some structure)
        base_matrix = np.full((J_base, K_base), -1, dtype=int)
        
        attempts = 0
        while True:
            attempts += 1
            temp_base = np.full((J_base, K_base), -1, dtype=int)
            
            # irregular degrees for columns (VN degree)
            # Typically for LDPC, we want some high degree VNs (e.g. 10-20%) and many degree 3 or 2
            for c in range(K_base):
                # Simple distribution
                rn = self.prng.random()
                if rn < 0.4: weight = 2
                elif rn < 0.85: weight = 3
                else: weight = 4 
                
                # Cap weight by available rows
                if weight > J_base: weight = J_base

                rows_chosen = self.prng.sample(range(J_base), weight)
                for r in rows_chosen:
                    temp_base[r, c] = self.prng.randint(0, b - 1)
            
            # Check row weights (CN degree) >= 2
            row_weights = np.sum(temp_base != -1, axis=1)
            if np.all(row_weights >= 2):
                base_matrix = temp_base
                break
            
            if attempts > 50:
                 # Force a valid matrix
                 for c in range(K_base):
                    # Ensure weight 2 at least
                    rows_chosen = self.prng.sample(range(J_base), 2)
                    for r in rows_chosen:
                        temp_base[r, c] = self.prng.randint(0, b - 1)
                 base_matrix = temp_base
                 break

        H_dok = dok_matrix((M, N), dtype=int)
        
        for r_base in range(J_base):
            for c_base in range(K_base):
                shift = base_matrix[r_base, c_base]
                if shift != -1:
                    for i in range(b):
                        row_idx = r_base * b + i
                        col_idx = c_base * b + (i + shift) % b
                        H_dok[row_idx, col_idx] = 1
        
        self.H = H_dok.tocsr()
        self.N = N
        self.M = M
        
        return self.H, self.N

    def get_check_node_indices(self) -> List[List[int]]:
        """
        Converts Sparse H matrix to list of lists structure.
        """
        if self.H is None: return []
        
        check_indices = []
        for i in range(self.M):
            row_start = self.H.indptr[i]
            row_end = self.H.indptr[i+1]
            cols = self.H.indices[row_start:row_end]
            check_indices.append(cols.tolist())
            
        return check_indices

    async def run(self, key: List[int], qber: float, oracle: ParityOracle):
        """
        Runs QC-LDPC Decoding using ldpc.BpDecoder.
        """
        if qber <= 0: qber = 0.00001
        
        orig_len = len(key)
        
        # Adaptive Construction
        H, N_adj = self.construct_H_matrix(orig_len, qber)
        
        if N_adj <= orig_len:
            if N_adj < orig_len:
                self.log(f"Truncating key from {orig_len} to {N_adj}.")
            processing_key = np.array(key[:N_adj], dtype=int)
        else:
             self.log(f"Key too short. Need {N_adj}, got {orig_len}")
             return key, 0, 0, 0

        
        # 1. Get Alice's Syndrome
        check_blocks = self.get_check_node_indices()
        self.channel_uses += 1
        alice_syndrome_list = await oracle.get_parities(check_blocks)
        self.bits_revealed += len(alice_syndrome_list)
        
        s_alice = np.array(alice_syndrome_list, dtype=int) # Shape (M,)

        # 2. Compute Bob's Syndrome for the current key
        s_bob = self.H.dot(processing_key) % 2
        
        # 3. Compute Difference Syndrome (Syndrome of the Error)
        s_diff = (s_alice + s_bob) % 2
        
        # 4. Decode
        # We model the error as a BS with probability qber
        try:
            # We use 'ps' (product-sum) for better accuracy or 'ms' (min-sum) for speed.
            # 'ps' is equivalent to tanh rule used in the manual implementation.
            decoder = ldpc.BpDecoder(
                self.H, 
                error_rate=qber, 
                bp_method='ps', 
                schedule='parallel', 
                max_iter=50
            )
            
            error_est = decoder.decode(s_diff)
            
            # Check if decoding converged/satisfied syndrome
            s_est = self.H.dot(error_est) % 2
            if not np.array_equal(s_est, s_diff):
                self.log("Decoder failed to converge to valid syndrome.")
                # We can fallback or just return what we have (best effort)
                # But if syndrome doesn't match, we likely have wrong codeword.
            else:
                 self.log(f"Decoder converged. Estimated {np.sum(error_est)} errors.")

            # 5. Correct Key
            # key_alice = key_bob + error
            corrected_key_arr = (processing_key + error_est) % 2
            
            flips = np.sum(np.abs(processing_key - corrected_key_arr))
            self.total_errors_corrected = flips
            
            return corrected_key_arr.tolist(), self.bits_revealed, self.total_errors_corrected, self.channel_uses

        except Exception as e:
            self.log(f"LDPC library error: {e}")
            return processing_key.tolist(), self.bits_revealed, 0, self.channel_uses
