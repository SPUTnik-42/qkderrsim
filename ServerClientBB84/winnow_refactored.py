import math
from typing import List, Protocol, Any
from prng import PRNG
import asyncio
import random

class ParityOracle(Protocol):
    async def get_parity(self, indices: List[int]) -> int:
        ...
    
    async def get_parities(self, blocks: List[List[int]]) -> List[int]:
        ...

class WinnowClientProtocol:
    """
    Winnow Error Correction Protocol for QKD.
    Uses Hamming codes to find and correct errors, reducing the number of communication rounds
    compared to Cascade.
    """
    def __init__(self, num_passes=6, verbose=True):
        self.num_passes = num_passes
        self.verbose = verbose
        
        # Global variables
        self.N = 0
        self.Q_int = 0
        
        # Metrics
        self.bits_revealed = 0
        self.total_errors_corrected = 0
        self.channel_uses = 0
        self.oracle = None

    def log(self, message):
        if self.verbose:
            print(message)

    def _deterministic_shuffle(self, data, seed):
        rng = PRNG(seed)
        shuffled_data = list(data)
        rng.shuffle(shuffled_data)
        return shuffled_data

    def calculate_local_parity(self, key_int: int, key_indices: List[int]) -> int:
        p_val = 0
        for idx in key_indices:
            shift = self.N - 1 - idx
            bit = (key_int >> shift) & 1
            p_val = p_val ^ bit
        return p_val

    def calculate_local_parities(self, key_int: int, blocks: List[List[int]]) -> List[int]:
        return [self.calculate_local_parity(key_int, block) for block in blocks]

    async def run(self, Q_key: List[int], qber: float, oracle: ParityOracle):
        self.log(f"Starting Winnow Protocol (Client) with {self.num_passes} passes.")
        if self.verbose:
            preview = "".join(map(str, Q_key[:50]))
            self.log(f"Initial Key (First 50 bits): {preview}...")
            
        self.oracle = oracle
        
        Q_str = "".join(map(str, Q_key))
        self.N = len(Q_key)
        self.Q_int = int(Q_str, 2)
        
        if qber is None or qber <= 0:
            qber = 0.01
            
        self.bits_revealed = 0
        self.total_errors_corrected = 0
        self.channel_uses = 0
        
        key_index_list = list(range(self.N))
        
        for i in range(1, self.num_passes + 1):
            self.log(f"\n--- Pass {i} ---")
            
            if i > 1:
                key_index_list = self._deterministic_shuffle(key_index_list, seed=i)
                self.log(f"  Shuffled key indices.")

            if i == 1:
                # Choose k such that the probability of >1 error is small.
                k_float = 0.5 / qber if qber > 0 else 8
                k = max(4, 2 ** math.floor(math.log2(max(1, k_float))))
            else:
                k = k * 2
                
            self.log(f"  Block size k: {k}")

            block_index_lists = []
            num_blocks = math.ceil(self.N / k)
            for l in range(1, num_blocks + 1):
                start_idx = (l - 1) * k
                end_idx = min(l * k, self.N)
                block = key_index_list[start_idx : end_idx]
                block_index_lists.append(block)

            # 1. Exchange overall parities
            self.channel_uses += 1
            alice_parity_list = await self.oracle.get_parities(block_index_lists)
            self.bits_revealed += len(alice_parity_list)
            
            bob_parity_list = self.calculate_local_parities(self.Q_int, block_index_lists)
            
            bad_blocks = []
            for j in range(len(alice_parity_list)):
                if alice_parity_list[j] != bob_parity_list[j]:
                    bad_blocks.append(j)
                    
            self.log(f"  Blocks with odd parity errors: {len(bad_blocks)} / {len(block_index_lists)}")

            # 2. For bad blocks, compute and exchange Hamming syndromes
            if bad_blocks:
                hamming_requests = []
                hamming_block_map = [] # Maps request index to (block_idx, syndrome_bit_idx)
                
                for b_idx in bad_blocks:
                    block = block_index_lists[b_idx]
                    block_len = len(block)
                    m = math.ceil(math.log2(block_len + 1))
                    
                    for j in range(m):
                        # Indices in this block where the j-th bit of (pos+1) is 1
                        syndrome_subset = []
                        for pos in range(block_len):
                            if ((pos + 1) >> j) & 1:
                                syndrome_subset.append(block[pos])
                        
                        if syndrome_subset:
                            hamming_requests.append(syndrome_subset)
                            hamming_block_map.append((b_idx, j))
                
                if hamming_requests:
                    self.channel_uses += 1
                    alice_syndromes = await self.oracle.get_parities(hamming_requests)
                    self.bits_revealed += len(alice_syndromes)
                    
                    bob_syndromes = self.calculate_local_parities(self.Q_int, hamming_requests)
                    
                    # Reconstruct S_diff for each bad block
                    s_diff_map = {b_idx: 0 for b_idx in bad_blocks}
                    
                    for req_idx, (b_idx, j) in enumerate(hamming_block_map):
                        a_syn = alice_syndromes[req_idx]
                        b_syn = bob_syndromes[req_idx]
                        if a_syn != b_syn:
                            s_diff_map[b_idx] |= (1 << j)
                            
                    # Correct errors
                    for b_idx, s_diff in s_diff_map.items():
                        if s_diff > 0:
                            error_pos = s_diff - 1
                            block = block_index_lists[b_idx]
                            if error_pos < len(block):
                                bit_flip_index = block[error_pos]
                                
                                # Flip bit
                                mask = 1 << (self.N - bit_flip_index - 1)
                                self.Q_int = self.Q_int ^ mask
                                self.total_errors_corrected += 1
                                
                                if self.verbose:
                                    self.log(f"    [WINNOW] Flipped bit at index {bit_flip_index} (block pos {error_pos})")
                            else:
                                if self.verbose:
                                    self.log(f"    [WINNOW] Syndrome {s_diff} out of bounds for block length {len(block)}. Multiple errors likely.")

        corrected_key_str = format(self.Q_int, f'0{self.N}b')
        corrected_key = [int(b) for b in corrected_key_str]
        return corrected_key, self.bits_revealed, self.total_errors_corrected, self.channel_uses

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
        print("--- Winnow Protocol Standalone Test ---")
        N = 1000
        qber = 0.03 # 3% error rate
        
        # 1. Generate Alice's Key (Correct)
        alice_key = [random.randint(0, 1) for _ in range(N)]
        
        # 2. Generate Bob's Key (Noisy)
        bob_key = list(alice_key)
        errors = 0
        for i in range(N):
            if random.random() < qber:
                bob_key[i] = 1 - bob_key[i]
                errors += 1
        
        print(f"Key Length: {N}")
        print(f"Initial Errors: {errors} (QBER: {errors/N:.2%})")
        
        # 3. Setup
        oracle = MockOracle(alice_key)
        winnow = WinnowClientProtocol(num_passes=4, verbose=True)
        
        # 4. Run Reconciliation
        corrected_key, revealed, corrected, rounds = await winnow.run(bob_key, qber, oracle)
        
        # 5. Verify
        final_errors = sum(1 for a, b in zip(alice_key, corrected_key) if a != b)
        
        print("\n--- Results ---")
        print(f"Bits Revealed: {revealed}")
        print(f"Errors Corrected: {corrected}")
        print(f"Remaining Errors: {final_errors}")
        print(f"Final Match: {100 * (N - final_errors) / N:.2f}%")
        
        if final_errors == 0:
            print("SUCCESS: Key perfectly reconciled!")
        else:
            print("FAILURE: Validation failed.")

    asyncio.run(main())
