import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix, dok_matrix
import ldpc
from prng import PRNG
from typing import List, Tuple, Optional
from cascade_refactored import ParityOracle
import math
import asyncio

class QCLDPCv3ClientProtocol:
    """
    Implements QC-LDPC error correction using the 'ldpc' (LDPCv2) library.
    Implements Direct Rate Adaptation Strategy.
    
    Strategy:
    1. Start with an optimistic high rate (less redundancy).
    2. If decoding fails, decrease the rate (add more redundancy) by a fixed step.
    """
    def __init__(self, verbose=True, rate=None, seed=None):
        self.verbose = verbose
        self.prng = PRNG(seed)
        # self.initial_rate = rate # Not really used in direct rate strategy if we start from capacity
        
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

    def _get_theoretical_max_rate(self, qber: float) -> float:
        """Calculates 1 - H(p). Theoretical max code rate."""
        if qber <= 0: return 1.0
        if qber >= 0.5: return 0.0
        h_p = -qber * math.log2(qber) - (1-qber) * math.log2(1-qber)
        return 1.0 - h_p

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

    def update_H_matrix(self, n_key: int, target_code_rate: float) -> Tuple[List[List[int]], int]:
        """
        Updates (extends) the H matrix to meet the target code rate.
        Returns:
            new_check_blocks: List of list of indices for the NEW checks only.
            N_valid: The valid block length.
        """
        K_base = 24 
        
        # J_base calculation
        # J = ceil( K * (1 - R) )
        target_J = math.ceil(K_base * (1.0 - target_code_rate))
        if target_J < 3: target_J = 3
        if target_J >= K_base: target_J = K_base - 1

        # Calculate b (lifting factor)
        # Only on first run
        if self.b == 0:
            # Use ceil to ensure we cover the whole key (requires padding later)
            self.b = math.ceil(n_key / K_base)
            if self.b < 1: self.b = 1
        
        # N should match existing or be set
        current_N = K_base * self.b
        
        # If we need more rows than we have
        if target_J > self.J_total:
            # Calculate how many rows to add
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
        Runs QC-LDPC Decoding with Direct Rate Adaptation.
        """
        if qber <= 0: qber = 0.00001
        
        orig_len = len(key)
        self.b = 0 # Reset block size calc
        self.J_total = 0
        self.base_matrix = None
        self.bits_revealed = 0
        self.channel_uses = 0
        
        # Direct Rate Parameters
        start_rate_offset = 0.05
        rate_step = 0.05
        min_rate = 0.1
        
        # 1. Determine Initial Rate
        theory_max = self._get_theoretical_max_rate(qber)
        current_rate = theory_max - start_rate_offset
        if current_rate > 0.95: current_rate = 0.95
        
        collected_syndrome = np.array([], dtype=int)
        processing_key = None
        
        while current_rate >= min_rate:
            self.log(f"--- LDPC Rate Pass: Rate {current_rate:.3f} ---")
            
            new_checks, N_valid = self.update_H_matrix(orig_len, current_rate)
            
            if processing_key is None:
                # Handle Padding if necessary (N_valid should cover orig_len)
                if N_valid > orig_len:
                     padding = N_valid - orig_len
                     processing_key = np.concatenate([key, np.zeros(padding, dtype=int)]).astype(int)
                else:
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
                # Rate mismatch or sync error, decrease rate to trigger new matrix regen if needed
                current_rate -= rate_step
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
                    
                    # --- CRITICAL VERIFICATION Step ---
                    # Protects against undetected errors (converging to wrong codeword)
                    candidate_key_est = (processing_key + error_est) % 2
                    
                    num_checks = 5 # Confidence level
                    verify_calls = []
                    for _ in range(num_checks):
                         # Sample ~50% of the key bits
                         subset_size = len(candidate_key_est) // 2
                         idxs = self.prng.sample(range(len(candidate_key_est)), subset_size)
                         verify_calls.append(idxs)
                    
                    # Cost of verification
                    self.bits_revealed += num_checks
                    self.channel_uses += 1
                    alice_parities = await oracle.get_parities(verify_calls)
                    
                    # Compare
                    match = True
                    for i, idxs in enumerate(verify_calls):
                        bob_p = np.sum(candidate_key_est[idxs]) % 2
                        if bob_p != alice_parities[i]:
                            match = False
                            break
                    
                    if match:
                        self.log(f"SUCCESS: Decoding converged and Verified at Rate {current_rate:.3f}.")
                        
                        # Strip padding to get back to original length
                        final_key = candidate_key_est[:orig_len]
                        
                        # Calculate errors corrected (approximate, since we don't have Alice's clean key here,
                        # but we can return the flips relative to Bob's input)
                        processing_key_orig = processing_key[:orig_len]
                        flips = np.sum(np.abs(processing_key_orig - final_key))
                        self.total_errors_corrected = flips
                        
                        return final_key.tolist(), self.bits_revealed, self.total_errors_corrected, self.channel_uses
                    else:
                        self.log(f"Verification Failed (Right Syndrome, Wrong Key) at Rate {current_rate:.3f}.")

                else:
                    self.log(f"Decoding failed (Syndrome mismatch) at Rate {current_rate:.3f}.")
                    
            except Exception as e:
                self.log(f"Decoder error: {e}")
                
            # Decrease and retry
            current_rate -= rate_step
        
        self.log("FAILURE: Min rate reached. Returning raw key.")
        return key, self.bits_revealed, 0, self.channel_uses

if __name__ == "__main__":
    
    class MockOracle:
        def __init__(self, key): 
            self.key = np.array(key, dtype=int)
            # Pad key in oracle to match protocol's padding logic
            orig_len = len(key)
            K_base = 24
            b = math.ceil(orig_len / K_base)
            if b < 1: b = 1
            pad_len = (b * K_base) - orig_len
            if pad_len > 0:
                self.key = np.concatenate([self.key, np.zeros(pad_len, dtype=int)])

        async def get_parities(self, blocks):
            res = []
            for b_idx in blocks:
                res.append(np.sum(self.key[b_idx]) % 2)
            return res

    async def main():
        N = 2000
        prng = PRNG(42)
        alice = [prng.randint(0,1) for _ in range(N)]
        
        qber = 0.2
        bob = list(alice)
        errs = 0
        h_p = -qber * math.log2(qber) - (1-qber) * math.log2(1-qber)
        for i in range(N):
            if prng.random() < qber:
                bob[i] = 1 - bob[i]
                errs += 1
        
        print(f"Direct Rate QC-LDPC Test. Errors: {errs} ({errs/N:.2%})")
        
        prot = QCLDPCv3ClientProtocol(verbose=True) 
        oracle = MockOracle(alice)
        
        res, bits, cor, uses = await prot.run(bob, qber, oracle)
        
        print(f"Corrected: {cor} out of Error:{errs}")
        print(f"Matches Alice? {np.array_equal(res, alice)}")
        print(f"Final Info Bits Revealed: {bits}")
        print(f"Efficiency: {bits/(N*h_p) :.2f}")

    asyncio.run(main())
# pick an h parity check matrix fix it and then pick a code word you insert errors randomly in the code word (medium amount of errors)
# errerounous message, from it apply the technique (bit flipping, syndrome calculation, belief propagation) to try to recover the original code word.
# what is the performance matrix 


"""
suppose k , from k get n length code word from ldpc, 
"""