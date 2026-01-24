
import math
import random
import asyncio
from typing import List, Protocol, Any

class ParityOracle(Protocol):
    async def get_parity(self, indices: List[int]) -> int:
        ...
    
    async def get_parities(self, blocks: List[List[int]]) -> List[int]:
        ...

class CascadeClientProtocol:
    def __init__(self, num_passes=4, verbose=True):
        self.num_passes = num_passes
        self.verbose = verbose
        
        # Global variables
        self.N = 0
        self.Q_int = 0
        self.net_block_index_lists = {}
        
        # Metrics
        self.bits_revealed = 0
        self.total_errors_corrected = 0
        self.channel_uses = 0
        self.oracle = None

    def log(self, message):
        if self.verbose:
            print(message)

    def _deterministic_shuffle(self, data, seed):
        rng = random.Random(seed)
        shuffled_data = list(data)
        rng.shuffle(shuffled_data)
        return shuffled_data

    def calculate_local_parity(self, key_int: int, key_indices: List[int]) -> int:
        p_val = 0
        for idx in key_indices:
            # key_int has bit 0 at LSB. 
            # If we assume key[0] is MSB, then we need to access N - 1 - idx
            # If we assume key[0] is LSB, then we access idx
            # The original code did: (Z_int >> (N - 1 - idx))
            # I will preserve that logic.
            shift = self.N - 1 - idx
            bit = (key_int >> shift) & 1
            p_val = p_val ^ bit
        return p_val

    def calculate_local_parities(self, key_int: int, blocks: List[List[int]]) -> List[int]:
        return [self.calculate_local_parity(key_int, block) for block in blocks]

    async def binary(self, block_index_list, pass_num):
        """
        Algorithm 2: BINARY (Client Side)
        """
        # self.log(f"  [BINARY] Starting binary search on block of size {len(block_index_list)}")
        
        beg = 0
        end = len(block_index_list) - 1
        
        while beg < end:
            mid = (beg + end) // 2
            
            # Python slicing is [start:end+1] to include end
            check_list_1 = block_index_list[beg : mid + 1]
            check_list_2 = block_index_list[mid + 1 : end + 1]
            
            # Bob (Client) sends check_list_1 to Alice (Server) -> Channel Use
            self.channel_uses += 1
            
            # Ask Alice for parity
            alice_parity = await self.oracle.get_parity(check_list_1)
            self.bits_revealed += 1
            
            # Compute local parity
            bob_parity = self.calculate_local_parity(self.Q_int, check_list_1)
            
            if bob_parity != alice_parity:
                if self.verbose:
                    self.log(f"    [BINARY] Mismatch in subset ({len(check_list_1)} bits). Searching left.")
                end = mid
                # Append check_list_2 to net_block_index_lists[pass_num]
                if pass_num not in self.net_block_index_lists:
                    self.net_block_index_lists[pass_num] = []
                self.net_block_index_lists[pass_num].append(check_list_2)
            else:
                if self.verbose:
                    self.log(f"    [BINARY] Match in subset. Error in right half ({len(check_list_2)} bits). Searching right.")
                beg = mid + 1
                # Append check_list_1 to net_block_index_lists[pass_num]
                if pass_num not in self.net_block_index_lists:
                    self.net_block_index_lists[pass_num] = []
                self.net_block_index_lists[pass_num].append(check_list_1)
        
        bit_flip_index = block_index_list[beg]
        if self.verbose:
             current_bit = (self.Q_int >> (self.N - 1 - bit_flip_index)) & 1
             self.log(f"  [FIXED] Found error at index {bit_flip_index}. Flipping bit {current_bit} -> {1-current_bit}")
        return bit_flip_index

    async def run(self, Q_key: List[int], qber: float, oracle: ParityOracle):
        """
        The Cascade Protocol Algorithm (Client Logic)
        """
        self.log(f"Starting Cascade Protocol (Client) with {self.num_passes} passes.")
        if self.verbose:
            preview = "".join(map(str, Q_key[:50]))
            self.log(f"Initial Key (First 50 bits): {preview}...")
        
        self.oracle = oracle
        
        Q_str = "".join(map(str, Q_key))
        self.N = len(Q_key)
        self.Q_int = int(Q_str, 2)
        self.net_block_index_lists = {}

        if qber is None:
            raise ValueError("Missing required parameter: qber must be provided.")
        
        # Reset Metrics
        self.bits_revealed = 0
        self.total_errors_corrected = 0
        self.channel_uses = 0
        
        key_index_list = list(range(self.N))
        
        for i in range(1, self.num_passes + 1):
            self.log(f"\n--- Pass {i} ---")
            
            if i > 1:
                key_index_list = self._deterministic_shuffle(key_index_list, seed=i) # Assuming Alice does same? YES.
                self.log(f"  Shuffled key indices.")

            if i == 1:
                k = int(0.73 / qber if 1 > qber > 0  else 9)
            else:
                k = k * (2 ** (i - 1))
            
            self.log(f"  Block size k: {k}")

            block_index_lists = []
            if i not in self.net_block_index_lists:
                self.net_block_index_lists[i] = []

            # Partition indices into blocks
            num_blocks = math.ceil(self.N / k)
            for l in range(1, num_blocks + 1):
                start_idx = (l - 1) * k
                end_idx = min(l * k, self.N)
                block = key_index_list[start_idx : end_idx]
                block_index_lists.append(block)

            # Bob sends all block definitions (implicitly or explicitly) to Alice.
            # In 'Parity API', he sends lists of indices.
            # "Bob sends block_index_lists to Alice" -> Channel Use (Bulk send)
            self.channel_uses += 1
            
            # API Call: Get parities for ALL blocks
            alice_parity_list = await self.oracle.get_parities(block_index_lists)
            self.bits_revealed += len(alice_parity_list)
            
            # Compute local parities
            bob_parity_list = self.calculate_local_parities(self.Q_int, block_index_lists)
            
            # Find blocks with odd error parity
            odd_error_parity_block_index_list = [] 
            for j in range(len(alice_parity_list)):
                if alice_parity_list[j] != bob_parity_list[j]:
                    odd_error_parity_block_index_list.append(j)
            
            self.log(f"  Blocks with odd parity errors: {len(odd_error_parity_block_index_list)} / {len(block_index_lists)}")

            # Add correct blocks to net_block_index_lists
            all_block_indices = set(range(len(block_index_lists)))
            correct_block_indices = all_block_indices - set(odd_error_parity_block_index_list)
            for j in correct_block_indices:
                self.net_block_index_lists[i].append(block_index_lists[j])

            net_index_list_for_bit_flip = []
            complete_index_list_for_bit_flip = []
            
            # Process bad blocks
            for j in odd_error_parity_block_index_list:
                self.log(f"  Correcting error in block {j}...")
                bit_flip_index = await self.binary(block_index_lists[j], i)
                
                # Flip bit in Q_int
                mask = 1 << (self.N - bit_flip_index - 1)
                self.Q_int = self.Q_int ^ mask
                self.total_errors_corrected += 1
                
                net_index_list_for_bit_flip.append(bit_flip_index)
                complete_index_list_for_bit_flip.append(bit_flip_index)
            
            # Cascade / Backtracking (if i > 1)
            if i > 1:
                self.log("  Starting Cascade/Backtracking...")
                net_check_block_index_lists = []
                
                # Identify previous blocks containing the flipped bits
                for bit_flip_idx in net_index_list_for_bit_flip:
                    for m in range(i - 1, 0, -1): 
                         if m in self.net_block_index_lists:
                             for sub_list in self.net_block_index_lists[m]:
                                 if bit_flip_idx in sub_list:
                                     if sub_list not in net_check_block_index_lists:
                                         net_check_block_index_lists.append(sub_list)
                
                while len(net_check_block_index_lists) > 0:
                    check_block_index_list = min(net_check_block_index_lists, key=len)
                    
                    check_block_pass_num = -1
                    for m_key, lists in self.net_block_index_lists.items():
                        if check_block_index_list in lists:
                            check_block_pass_num = m_key
                            break
                    
                    net_check_block_index_lists.remove(check_block_index_list)
                    
                    parity_check = 0
                    for error_index in complete_index_list_for_bit_flip:
                        if error_index in check_block_index_list:
                            parity_check += 1
                    
                    if parity_check % 2 != 0:
                        self.log(f"    [CASCADE] Re-evaluating block from Pass {check_block_pass_num} (len {len(check_block_index_list)})")
                        
                        if check_block_pass_num != -1:
                            if check_block_index_list in self.net_block_index_lists[check_block_pass_num]:
                                self.net_block_index_lists[check_block_pass_num].remove(check_block_index_list)
                        
                        new_bit_flip_index = await self.binary(check_block_index_list, check_block_pass_num)
                        
                        mask = 1 << (self.N - new_bit_flip_index - 1)
                        self.Q_int = self.Q_int ^ mask
                        self.total_errors_corrected += 1
                        
                        complete_index_list_for_bit_flip.append(new_bit_flip_index)
                        
                        check_block_new_additions = []
                        
                        for m in range(i, 0, -1):
                            if m != check_block_pass_num:
                                if m in self.net_block_index_lists:
                                    for sub_list in self.net_block_index_lists[m]:
                                        if new_bit_flip_index in sub_list:
                                            check_block_new_additions.append(sub_list)
                        
                        if check_block_new_additions:
                            for new_add in check_block_new_additions:
                                if new_add not in net_check_block_index_lists:
                                    net_check_block_index_lists.append(new_add)

        corrected_key_str = format(self.Q_int, f'0{self.N}b')
        corrected_key = [int(b) for b in corrected_key_str]
        return corrected_key, self.bits_revealed, self.total_errors_corrected, self.channel_uses
