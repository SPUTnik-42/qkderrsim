import math
import asyncio
from typing import List, Protocol

class ParityOracle(Protocol):
    async def get_parity(self, indices: List[int]) -> int:
        ...
    async def get_parities(self, blocks: List[List[int]]) -> List[int]:
        ...

class PolarClientProtocol:
    """
    Interactive Polar Decoder for Quantum Key Distribution based on the paper:
    'Polar Codes for Quantum Key Distribution' by Nakassis A (2017).
    """
    def __init__(self, verbose: bool = True, u_fer_target: float = 0.01, c: float = 0.5):
        self.verbose = verbose
        self.u_fer = u_fer_target
        self.c = c
        self.unused_fer = u_fer_target
        self.peeks = 0
        self.oracle = None
        self.r_matrix = []
        self.bit_matrix = []

    def log(self, message):
        if self.verbose:
            print(message)

    async def run(self, Q_key: List[int], qber: float, oracle: ParityOracle):
        self.oracle = oracle
        self.N_input = len(Q_key)
        
        # Pad to power of 2
        self.n = max(0, math.ceil(math.log2(self.N_input))) if self.N_input > 0 else 0
        self.N = 2**self.n
        padded_key = Q_key + [0] * (self.N - self.N_input)
        
        self.log(f"Starting Polar Protocol (Client) with N={self.N}, target FER={self.u_fer}")
        
        self.unused_fer = self.u_fer
        self.peeks = 0

        self.r_matrix = [[0.0 for _ in range(self.N)] for _ in range(self.n + 1)]
        self.bit_matrix = [[-1 for _ in range(self.N)] for _ in range(self.n + 1)]

        # Initialize the channel side (column n) r-values
        if qber <= 0: qber = 0.01
        r_val = 1.0 - 2.0 * qber
        for i in range(self.N):
            if i >= self.N_input:
                self.r_matrix[self.n][i] = 1.0 # 0 error for padding
            else:
                self.r_matrix[self.n][i] = r_val if padded_key[i] == 0 else -r_val

        # Store bit indices for $u$ mapping to $x$ corresponding combinations
        self.row_indices = self._compute_G_rows()

        # SC Decoding step
        await self._decode_node(0, 0, self.n)

        # After decoding column 0, we can re-encode to find the corrected codeword
        corrected_u = [self.bit_matrix[0][i] for i in range(self.N)]
        corrected_codeword = self._encode_polar(corrected_u)

        # Return truncated key
        final_key = corrected_codeword[:self.N_input]
        
        # We don't accurately know total errors corrected trivially here, so we approximate
        total_errors_corrected = sum(1 for i in range(self.N_input) if final_key[i] != Q_key[i])
        
        return final_key, self.peeks, total_errors_corrected, self.peeks

    def _compute_G_rows(self) -> List[List[int]]:
        """
        Since u = x G and G = G_N is the polar transform matrix (which is symmetric G = G^{-1}),
        the k-th bit of u depends on a subset of bits of x. We compute the indices for each row.
        """
        rows = [[i] for i in range(self.N)]
        v = list(range(self.N))
        
        # We can simulate the polar transform to find dependencies
        deps = [{i} for i in range(self.N)]
        for i in range(self.n):
            step = 1 << i
            for j in range(0, self.N, 2 * step):
                for k in range(step):
                    # u_even = u_even + u_odd
                    deps[j + k] = deps[j + k].union(deps[j + k + step])
        
        return [sorted(list(deps[i])) for i in range(self.N)]

    def _f(self, r_X_star: float, r_Y_star: float) -> float:
        # r[X] = r[X*] * r[Y*]
        return r_X_star * r_Y_star

    def _g(self, r_X_star: float, r_Y_star: float, b_X: int) -> float:
        # Eq 2a & 2b:
        if b_X == 0:
            denom = 1.0 + r_X_star * r_Y_star
            if denom == 0.0: return 0.0
            return (r_X_star + r_Y_star) / denom
        else:
            denom = 1.0 - r_X_star * r_Y_star
            if denom == 0.0: return 0.0
            return (r_Y_star - r_X_star) / denom

    async def _decode_node(self, m: int, offset: int, stage: int):
        if stage == 0:
            # We reached column 0, use I_D logic to query or decide the bit
            r_val = self.r_matrix[0][offset]
            p0 = (1.0 + r_val) / 2.0
            
            x = min(self.c, self.unused_fer / (self.N - offset))

            if p0 > 1.0 - x:
                self.bit_matrix[0][offset] = 0
                self.unused_fer -= (1.0 - p0)
            elif p0 < x:
                self.bit_matrix[0][offset] = 1
                self.unused_fer -= p0
            else:
                self.peeks += 1
                subset = [idx for idx in self.row_indices[offset] if idx < self.N_input]
                if not subset:
                    val = 0
                else:
                    val = await self.oracle.get_parity(subset)
                self.bit_matrix[0][offset] = val
            return

        half_length = 2**(stage - 1)
        
        # f-transform (left traversal)
        for i in range(half_length):
            idx1 = offset + i
            idx2 = offset + half_length + i
            r_X_star = self.r_matrix[stage][idx1]
            r_Y_star = self.r_matrix[stage][idx2]
            
            self.r_matrix[stage - 1][idx1] = self._f(r_X_star, r_Y_star)

        # Recurse left
        await self._decode_node(m + 1, offset, stage - 1)

        # g-transform (right traversal)
        for i in range(half_length):
            idx1 = offset + i
            idx2 = offset + half_length + i
            r_X_star = self.r_matrix[stage][idx1]
            r_Y_star = self.r_matrix[stage][idx2]
            b_X = self.bit_matrix[stage - 1][idx1]
            
            self.r_matrix[stage - 1][idx2] = self._g(r_X_star, r_Y_star, b_X)

        # Recurse right
        await self._decode_node(m + 1, offset + half_length, stage - 1)

        # Combine decisions up the tree
        for i in range(half_length):
            idx1 = offset + i
            idx2 = offset + half_length + i
            b_X = self.bit_matrix[stage - 1][idx1]
            b_Y = self.bit_matrix[stage - 1][idx2]
            
            self.bit_matrix[stage][idx1] = b_X ^ b_Y
            self.bit_matrix[stage][idx2] = b_Y

    def _encode_polar(self, u: List[int]) -> List[int]:
        v = list(u)
        for i in range(self.n):
            step = 1 << i
            for j in range(0, len(v), 2 * step):
                for k in range(step):
                    v[j + k] ^= v[j + k + step]
        return v

if __name__ == "__main__":
    class MockOracle:
        def __init__(self, key):
            self.key = key
        
        async def get_parity(self, indices: List[int]) -> int:
            p_val = 0
            for idx in indices:
                p_val ^= self.key[idx]
            return p_val
    
        async def get_parities(self, blocks: List[List[int]]) -> List[int]:
            return [await self.get_parity(block) for block in blocks]

    async def main():
        import random
        print("--- Polar Codes Protocol Standalone Test ---")
        N_input = 1000
        qber = 0.03 # 3% error rate
        
        # 1. Generate Alice's Key (Correct)
        alice_key = [random.randint(0, 1) for _ in range(N_input)]
        
        # 2. Generate Bob's Key (Noisy)
        bob_key = list(alice_key)
        errors = 0
        for i in range(N_input):
            if random.random() < qber:
                bob_key[i] = 1 - bob_key[i]
                errors += 1
        
        print(f"Key Length: {N_input}")
        print(f"Initial Errors: {errors} (QBER: {errors/N_input:.2%})")
        
        # 3. Setup
        oracle = MockOracle(alice_key)
        polar = PolarClientProtocol(verbose=True, u_fer_target=0.01)
        
        # 4. Run Reconciliation
        corrected_key, revealed, corrected, rounds = await polar.run(bob_key, qber, oracle)
        
        # 5. Verify
        final_errors = sum(1 for a, b in zip(alice_key, corrected_key) if a != b)
        
        print("\n--- Results ---")
        print(f"Bits Revealed: {revealed}")
        print(f"Estimated Errors Corrected: {corrected}")
        print(f"Remaining Errors: {final_errors}")
        print(f"Final Match: {100 * (N_input - final_errors) / N_input:.2f}%")
        
        if final_errors == 0:
            print("SUCCESS: Key perfectly reconciled!")
        else:
            print("FAILURE: Validation failed.")

    asyncio.run(main())
