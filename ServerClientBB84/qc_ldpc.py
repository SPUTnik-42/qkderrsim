import numpy as np
import math
from prng import PRNG
from typing import List, Tuple, Optional
import asyncio
from scipy.sparse import csr_matrix, dok_matrix
from cascade_refactored import ParityOracle

class QCLDPCClientProtocol:
    """
    Implements Quasi-Cyclic LDPC (QC-LDPC) error correction.
    Designed to be a drop-in replacement for CascadeClientProtocol. 
    Optimized for sparse matrix operations and irregular structure.
    Does NOT depend on external 'ldpc' library (Native Python/Scipy).
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
            print(f"[QC-LDPC-Native] {message}")

    def _calculate_adaptive_rate(self, qber: float) -> float:
        """
        Calculates optimal coding rate K/N based on QBER.
        R = 1 - (efficiency * H(qber))
        """
        if qber <= 0: return 0.9
        if qber >= 0.5: return 0.0
        
        # Binary Entropy
        h_p = -qber * math.log2(qber) - (1-qber) * math.log2(1-qber)
        
        inefficiency = 1.20 # 20% overhead
        syndrome_rate = h_p * inefficiency
        
        if syndrome_rate >= 0.95: 
            syndrome_rate = 1.0 
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
        # R = (K - J) / K = 1 - J/K => J/K = 1 - R
        J_base = math.ceil(K_base * (1.0 - target_code_rate)) 
        
        if J_base < 3: J_base = 3 
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
            for c in range(K_base):
                rn = self.prng.random()
                if rn < 0.4: weight = 2
                elif rn < 0.85: weight = 3
                else: weight = 4 
                
                if weight > J_base: weight = J_base

                rows_chosen = self.prng.sample(range(J_base), weight)
                for r in rows_chosen:
                    temp_base[r, c] = self.prng.randint(0, b - 1)
            
            # Validation: Block Cycle Check (ghetto PEG)
            # Just check if we created rows with 0 or 1 weight (useless checks)
            row_weights = np.sum(temp_base != -1, axis=1)
            if np.all(row_weights >= 2):
                base_matrix = temp_base
                break
            
            if attempts > 50:
                 # Force a valid matrix
                 for c in range(K_base):
                    rows_chosen = self.prng.sample(range(J_base), 2)
                    for r in rows_chosen:
                        temp_base[r, c] = self.prng.randint(0, b - 1)
                 base_matrix = temp_base
                 break

        # 3. Expand to Sparse H
        # Use dok_matrix for efficient construction
        H_dok = dok_matrix((M, N), dtype=int)
        
        for r_base in range(J_base):
            for c_base in range(K_base):
                shift = base_matrix[r_base, c_base]
                if shift != -1:
                    # Add circulant identity shifted by 'shift'
                    # The circulant P has P[i, (i+shift)%b] = 1
                    for i in range(b):
                        row_idx = r_base * b + i
                        col_idx = c_base * b + (i + shift) % b
                        H_dok[row_idx, col_idx] = 1
        
        # Convert to CSR for arithmetic efficiency
        self.H = H_dok.tocsr()
        self.N = N
        self.M = M
        
        return self.H, self.N

    def get_check_node_indices(self) -> List[List[int]]:
        """
        Converts Sparse H matrix to list of lists structure.
        Compatible with the Oracle API.
        """
        if self.H is None: return []
        
        check_indices = []
        # CSR makes this efficient: iterates rows
        for i in range(self.M):
            # Indices for row i are in self.H.indices[self.H.indptr[i]:self.H.indptr[i+1]]
            row_start = self.H.indptr[i]
            row_end = self.H.indptr[i+1]
            cols = self.H.indices[row_start:row_end]
            check_indices.append(cols.tolist())
            
        return check_indices

    async def run(self, key: List[int], qber: float, oracle: ParityOracle):
        """
        Runs QC-LDPC Decoding using Sparse Belief Propagation.
        Algorithm: Sum-Product (Log Domain)
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
        
        # Get Syndrome
        check_blocks = self.get_check_node_indices()
        self.channel_uses += 1
        alice_syndrome = await oracle.get_parities(check_blocks)
        self.bits_revealed +=  len(alice_syndrome)
        S = np.array(alice_syndrome, dtype=int) # Shape (M,)

        # --- Belief Propagation (Sparse) ---
        
        # 1. Initialization (LLR)
        L_const = np.log((1 - qber) / qber)
        # LLR_intrinsic: shape (N,)
        LLR_intrinsic = np.where(processing_key == 0, L_const, -L_const)
        
        # Initialize messages
        # We store messages aligned with the non-zero elements of H (edges)
        # H is CSR. data, indices, indptr.
        # num_edges = H.nnz
        num_edges = self.H.nnz
        
        # M_v_c (Variable to Check) messages: initially LLR_intrinsic[v] for each edge
        # We need to map edge index k -> column index j
        # In CSR, 'indices' array holds column indices.
        col_indices_per_edge = self.H.indices
        
        # Initialize M_v_c with intrinsic LLR of the connected variable node
        M_v_c = LLR_intrinsic[col_indices_per_edge] 
        
        # M_c_v (Check to Variable) messages: initially 0
        M_c_v = np.zeros(num_edges)
        
        max_iter = 50
        success = False
        final_key = processing_key.copy()
        
        for iteration in range(max_iter):
            # Step C: Check Node Update (CN -> VN)
            # For each row (Check Node), compute product of tanh(M_v_c/2) for all neighbors
            # then exclude the target edge.
            
            T_v_c = np.tanh(M_v_c / 2.0)
            T_v_c = np.clip(T_v_c, -0.999999, 0.999999) # Numerical stability
            
            # Efficient Row Product using reduceat on CSR structure
            # indptr tells us where each row starts/ends.
            # np.multiply.reduceat(array, indices) computes reductions at slices.
            # We want reduce at [0, r1, r2...] basically self.H.indptr[:-1]
            
            # Note: reduceat requires indices to be strictly increasing? 
            # indptr is monotonic.
            full_row_products = np.multiply.reduceat(T_v_c, self.H.indptr[:-1])
            
            # Now we need to map these row products back to edges.
            # repeat row_product[i] for degree[i] times.
            row_lengths = np.diff(self.H.indptr)
            expanded_row_products = np.repeat(full_row_products, row_lengths)
            
            # "Exclude target edge" -> Divide by the T_v_c of that edge
            # T_except_self = Total / T_self
            T_c_v_raw = expanded_row_products / T_v_c
            T_c_v_raw = np.clip(T_c_v_raw, -0.999999, 0.999999)
            
            # Apply Syndrome constraint: (-1)^Sc
            # Expand S to edges
            expanded_S = np.repeat(S, row_lengths)
            parity_sign = np.where(expanded_S == 1, -1.0, 1.0)
            
            M_c_v = parity_sign * 2 * np.arctanh(T_c_v_raw)
            
            # Step B: Variable Node Update (VN -> CN) & Hard Decision
            # We need to sum M_c_v messages coming into each variable node v.
            # Since H is CSR (Row-major), identifying incoming cols is hard efficiently 
            # without converting to CSC (Column-major).
            
            # Optimization: 
            # L_total[v] = LLR_intrinsic[v] + Sum_{all c connected to v} M_c_v
            # M_v_c = L_total[v] - M_c_v (reverse edge)
            
            # To sum M_c_v per column efficiently:
            # We can use np.add.at -- fast unbuffered summation
            sum_incoming = np.zeros(self.N)
            np.add.at(sum_incoming, col_indices_per_edge, M_c_v)
            
            L_total = LLR_intrinsic + sum_incoming
            
            # Hard Decision
            current_estimation = np.where(L_total > 0, 0, 1)
            
            # Syndrome Check using efficient Sparse Multiply
            # H is CSR, current_estimation is vector. H.dot() is fast.
            S_check = self.H.dot(current_estimation) % 2
            
            if np.array_equal(S_check, S):
                success = True
                final_key = current_estimation
                if self.verbose:
                    self.log(f"Converged at iteration {iteration+1}.")
                break
                
            # Prepare M_v_c for next iteration
            # M_v_c[edge] = L_total[col[edge]] - M_c_v[edge]
            M_v_c = L_total[col_indices_per_edge] - M_c_v

        if not success:
            self.log("Max iterations reached.")
            final_key = np.where(L_total > 0, 0, 1)

        flips = np.sum(np.abs(processing_key - final_key))
        self.total_errors_corrected = flips
        
        return final_key.tolist(), self.bits_revealed, self.total_errors_corrected, self.channel_uses

