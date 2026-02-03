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
    Now supports Incremental Redundancy (Hybrid-ARQ) for correct efficiency calculation.
    """
    def __init__(self, verbose=True, rate=0.5, seed=None):
        self.verbose = verbose
        self.prng = PRNG(seed)
        self.initial_rate = rate 
        
        # State for Adaptive/Incremental Redundancy
        self.H: Optional[csr_matrix] = None
        self.base_matrix: Optional[np.ndarray] = None
        self.J_total: int = 0
        self.b: int = 0
        self.N: int = 0
        self.M: int = 0
        
        self.revealed_indices_count = 0 
        # Note: We track total bits requested across all attempts.
        
        self.bits_revealed = 0
        self.total_errors_corrected = 0
        self.channel_uses = 0

    def log(self, message):
        if self.verbose:
            print(f"[QC-LDPC-v2] {message}")

    def _calculate_adaptive_rate(self, qber: float, inefficiency: float = 1.05) -> float:
        """
        Calculates optimal coding rate based on QBER and inefficiency factor.
        R = 1 - (efficiency * H(qber))
        """
        if qber <= 0: return 0.95
        if qber >= 0.5: return 0.1 # Very low rate
        
        # Binary Entropy
        if qber == 0 or qber == 1:
            h_p = 0
        else:
            h_p = -qber * math.log2(qber) - (1-qber) * math.log2(1-qber)
        
        # Syndrome Rate = M/N = inefficiency * H(p)
        syndrome_rate = h_p * inefficiency
        
        # Robustness bounds
        if syndrome_rate >= 0.95: 
            syndrome_rate = 0.98
        elif syndrome_rate < 0.05: 
            syndrome_rate = 0.05
        
        code_rate = 1.0 - syndrome_rate
        return code_rate

    def _generate_base_rows(self, rows_to_add: int, K_base: int) -> np.ndarray:
        """
        Generates new rows for the base matrix.
        Ensures each row has at least weight 2.
        """
        new_rows = np.full((rows_to_add, K_base), -1, dtype=int)
        
        # Heuristic: fill rows to have weight ~7
        target_row_weight = 7 
        if target_row_weight > K_base: target_row_weight = K_base
        
        for r in range(rows_to_add):
            # Pick 'target_row_weight' random columns
            cols = self.prng.sample(range(K_base), target_row_weight)
            for c in cols:
                new_rows[r, c] = self.prng.randint(0, self.b - 1)
                
        return new_rows

    def update_H_matrix(self, n_key: int, qber: float, inefficiency: float) -> Tuple[List[List[int]], int]:
        """
        Updates (extends) the H matrix to meet the target inefficiency.
        Returns:
            new_check_blocks: List of list of indices for the NEW checks only.
            N_valid: The valid block length.
        """
        # 1. Determine Target Parameters
        if self.initial_rate == "adaptive" or self.initial_rate is None or self.initial_rate == "auto":
            target_code_rate = self._calculate_adaptive_rate(qber, inefficiency)
        else:
            target_code_rate = float(self.initial_rate)

        K_base = 24 
        
        # J_base calculation
        # J = ceil( K * (1 - R) )
        target_J = math.ceil(K_base * (1.0 - target_code_rate))
        if target_J < 3: target_J = 3
        if target_J >= K_base: target_J = K_base - 1

        # Calculate b (lifting factor)
        # Only on first run
        if self.b == 0:
            self.b = n_key // K_base
            if self.b < 1: self.b = 1
        
        # N should match existing or be set
        current_N = K_base * self.b
        
        # If we need more rows than we have
        if target_J > self.J_total:
            rows_to_add = target_J - self.J_total
            self.log(f"Extending H: J from {self.J_total} to {target_J} (Adding {rows_to_add} rows). Rate -> {target_code_rate:.3f}")
            
            # Generate new rows
            if self.base_matrix is None:
                # First time: Generate using the weighted column logic for better structure
                self.base_matrix = np.full((target_J, K_base), -1, dtype=int)
                
                temp_base = np.full((target_J, K_base), -1, dtype=int)
                
                # Column-based distribution for standard LDPC structure
                for c in range(K_base):
                    rn = self.prng.random()
                    if rn < 0.4: weight = 2
                    elif rn < 0.85: weight = 3
                    else: weight = 4 
                    if weight > target_J: weight = target_J
                    
                    rows_chosen = self.prng.sample(range(target_J), weight)
                    for r in rows_chosen:
                        temp_base[r, c] = self.prng.randint(0, self.b - 1)
                
                # Ensure row weights >= 2 (Safety check)
                for r in range(target_J):
                    if np.sum(temp_base[r] != -1) < 2:
                        # force add 2 entries
                        cols = self.prng.sample(range(K_base), 2)
                        for c in cols:
                            temp_base[r, c] = self.prng.randint(0, self.b - 1)
                            
                self.base_matrix = temp_base
                new_row_start_idx = 0
            
            else:
                # Extension steps
                extension = self._generate_base_rows(rows_to_add, K_base)
                self.base_matrix = np.vstack([self.base_matrix, extension])
                new_row_start_idx = self.J_total

            # Update State
            self.J_total = target_J
            self.N = current_N
            self.M = self.J_total * self.b
            
            # Rebuild Sparse H from base_matrix (Full Rebuild is fast enough)
            self._rebuild_csr_H(K_base)
            
            # Extract ONLY the new check blocks to request
            # Rows from (new_row_start_idx * b) to (target_J * b)
            new_checks = []
            start_row_abs = new_row_start_idx * self.b
            
            for i in range(start_row_abs, self.M):
                row_start = self.H.indptr[i]
                row_end = self.H.indptr[i+1]
                cols = self.H.indices[row_start:row_end]
                new_checks.append(cols.tolist())
                
            return new_checks, self.N
            
        else:
            # No extension needed
            return [], self.N

    def _rebuild_csr_H(self, K_base):
        J = self.base_matrix.shape[0]
        rows = []
        cols = []
        
        for r_base in range(J):
            for c_base in range(K_base):
                 shift = self.base_matrix[r_base, c_base]
                 if shift != -1:
                     for i in range(self.b):
                         row_idx = r_base * self.b + i
                         col_idx = c_base * self.b + (i + shift) % self.b
                         rows.append(row_idx)
                         cols.append(col_idx)
        
        data = np.ones(len(rows), dtype=int)
        self.H = csr_matrix((data, (rows, cols)), shape=(J * self.b, K_base * self.b))

    async def run(self, key: List[int], qber: float, oracle: ParityOracle):
        """
        Runs QC-LDPC Decoding with Incremental Redundancy.
        """
        if qber <= 0: qber = 0.00001
        
        orig_len = len(key)
        self.b = 0 # Reset block size calc
        self.J_total = 0
        self.base_matrix = None
        self.bits_revealed = 0
        self.channel_uses = 0
        
        # 1. Optimistic Start (Inefficiency ~1.05)
        current_inefficiency = 1.05
        max_inefficiency = 2.0 
        
        collected_syndrome = np.array([], dtype=int)
        processing_key = None
        
        while current_inefficiency <= max_inefficiency:
            self.log(f"--- LDPC IR Pass: Inefficiency {current_inefficiency:.2f} ---")
            
            new_checks, N_valid = self.update_H_matrix(orig_len, qber, current_inefficiency)
            
            if processing_key is None:
                if N_valid > orig_len:
                     # This shouldn't happen with b = floor(len/24)
                     self.log("Error: N_valid > orig_len")
                     return key, 0, 0, 0
                processing_key = np.array(key[:N_valid], dtype=int)

            if new_checks:
                self.log(f"Requesting {len(new_checks)} new parity bits.")
                self.channel_uses += 1
                new_syndrome_list = await oracle.get_parities(new_checks)
                self.bits_revealed += len(new_syndrome_list)
                
                # Append to collected syndrome
                new_s = np.array(new_syndrome_list, dtype=int)
                collected_syndrome = np.concatenate([collected_syndrome, new_s])
            else:
                self.log("No new checks generated (Already at max J?).")
            
            # --- Try Decoding ---
            
            # 1. Calculate local syndrome (current H)
            s_bob = self.H.dot(processing_key) % 2
            
            # 2. Calculate Difference
            if len(collected_syndrome) != s_bob.shape[0]:
                self.log(f"Shape Mismatch: Collected {len(collected_syndrome)} vs M {s_bob.shape[0]}")
                # Fallback: maybe just assume we failed or fix logic
                current_inefficiency += 0.05
                continue
                
            s_diff = (collected_syndrome + s_bob) % 2
            
            # 3. Decode
            try:
                decoder = ldpc.BpDecoder(
                    self.H, 
                    error_rate=qber, 
                    bp_method='ps', 
                    schedule='parallel', 
                    max_iter=50
                )
                error_est = decoder.decode(s_diff)
                
                # 4. Check Validity (Strict Syndrome Check)
                s_est = self.H.dot(error_est) % 2
                if np.array_equal(s_est, s_diff):
                    self.log(f"SUCCESS: Decoding converged at inefficiency {current_inefficiency:.2f}.")
                    
                    corrected_key_arr = (processing_key + error_est) % 2
                    flips = np.sum(np.abs(processing_key - corrected_key_arr))
                    self.total_errors_corrected = flips
                    
                    return corrected_key_arr.tolist(), self.bits_revealed, self.total_errors_corrected, self.channel_uses
                else:
                    self.log("Decoding failed check. Increasing redundancy.")
                    
            except Exception as e:
                self.log(f"Decoder error: {e}")
                
            # Increase and retry
            current_inefficiency += 0.05
        
        self.log("FAILURE: Max redundancy reached. Returning raw key.")
        return processing_key.tolist(), self.bits_revealed, 0, self.channel_uses
