import math
import random

class CascadeProtocol:
    def __init__(self, num_passes=4,  verbose=True): #initial_block_size=8,
        self.num_passes = num_passes
        # self.initial_block_size = initial_block_size
        self.verbose = verbose
        
        # Global variables as defined in the algorithm
        self.N = 0
        self.P_int = 0
        self.Q_int = 0
        self.net_block_index_lists = {}
        
        # Metrics
        self.bits_revealed = 0
        self.total_errors_corrected = 0
        self.channel_uses = 0

    def log(self, message):
        if self.verbose:
            print(message)

    def _deterministic_shuffle(self, data, seed):
        """
        Deterministic shuffle to simulate random_shuffle() in the algorithm.
        """
        rng = random.Random(seed)
        shuffled_data = list(data)
        rng.shuffle(shuffled_data)
        return shuffled_data

    def parity(self, Z_int, key_index_lists):
        """
        Algorithm 1: Parity
        """
        # If input is a single list, wrap it in a list to treat it uniformly
        # The algorithm implies key_index_lists is a list of lists.
        # However, binary() calls it with a single list.
        # We handle both cases to match the polymorphic nature of the pseudo-code.
        is_single_list = False
        if not key_index_lists:
            return []
        if isinstance(key_index_lists[0], int):
            key_index_lists = [key_index_lists]
            is_single_list = True

        N_l = len(key_index_lists)
        parity_list = [0] * N_l
        
        for i in range(N_l):
            # In Python, we don't need to invert indices (N - 1 - index) unless 
            # specifically matching a hardware implementation direction.
            # The algorithm says: key_index_lists[i][j] <- N - 1 - key_index_lists[i][j]
            # This reverses bit significance order (LSB vs MSB). 
            # We will follow the algorithm strictly.
            
            # Create a copy to avoid modifying the original list in place during calculation if needed,
            # but the algorithm modifies it.
            # "key_index_lists[i][j] <- N - 1 - ..." implies modification.
            # However, Z_int >> k assumes k is the shift amount.
            # If we invert k, we might be looking at the wrong bit if Z_int is standard integer.
            # Standard python int: bit 0 is LSB. 
            # If the algorithm considers index 0 as left-most (MSB), this inversion maps it to LSB.
            
            current_indices = []
            for idx in key_index_lists[i]:
                # implementing: N - 1 - index
                current_indices.append(self.N - 1 - idx)
            
            p_val = 0
            for k in current_indices:
                bit = (Z_int >> k) & 1
                p_val = p_val ^ bit # XOR is equivalent to parity addition
            
            parity_list[i] = p_val
            
            # Increment bits revealed for every parity bit calculated/exposed
            #self.bits_revealed += 1

        if is_single_list:
            return parity_list[0]
        return parity_list

    def binary(self, block_index_list, pass_num):
        """
        Algorithm 2: BINARY
        """
        self.log(f"  [BINARY] Starting binary search on block of size {len(block_index_list)}")
        
        beg = 0
        end = len(block_index_list) - 1
        
        while beg < end:
            mid = (beg + end) // 2
            
            # Python slicing is [start:end+1] to include end
            check_list_1 = block_index_list[beg : mid + 1]
            check_list_2 = block_index_list[mid + 1 : end + 1]
            
            # "Bob sends check_list_1 to Alice" -> Channel Use
            self.channel_uses += 1
            
            # "Alice computes Alice_parity"
            Alice_parity = self.parity(self.P_int, check_list_1)
            #####!!!!!!
            self.bits_revealed += 1
            
            # "Alice sends Alice_parity to Bob" -> Part of the ask-reply round trip
            
            Bob_parity = self.parity(self.Q_int, check_list_1)
            
            if Bob_parity != Alice_parity:
                self.log(f"    [BINARY] Parity mismatch in left half ({beg}-{mid}). Recursing left.")
                end = mid
                # Append check_list_2 to net_block_index_lists[pass_num]
                if pass_num not in self.net_block_index_lists:
                    self.net_block_index_lists[pass_num] = []
                self.net_block_index_lists[pass_num].append(check_list_2)
            else:
                self.log(f"    [BINARY] Parity match in left half. Error in right half ({mid+1}-{end}). Recursing right.")
                beg = mid + 1
                # Append check_list_1 to net_block_index_lists[pass_num]
                if pass_num not in self.net_block_index_lists:
                    self.net_block_index_lists[pass_num] = []
                self.net_block_index_lists[pass_num].append(check_list_1)
        
        bit_flip_index = block_index_list[beg]
        self.log(f"  [BINARY] Found error at index: {bit_flip_index}")
        return bit_flip_index

    def run(self, P_key, Q_key, qber):
        """
        The Cascade Protocol Algorithm
        Arguments:
            P_key: Alice's key (List[int])
            Q_key: Bob's key (List[int])
            qber: Quantum Bit Error Rate (float)
        """
        self.log(f"Starting Cascade Protocol with {self.num_passes} passes.")
        
        # Inputs must be list of ints
        P_str = "".join(map(str, P_key))
        Q_str = "".join(map(str, Q_key))

        # Reset Global Variables
        self.N = len(P_key)
        self.P_int = int(P_str, 2)
        self.Q_int = int(Q_str, 2)
        self.net_block_index_lists = {} # Using dict for sparse array behavior

        # errors = sum(1 for a, b in zip(P_str, Q_str) if a != b)
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
                key_index_list = self._deterministic_shuffle(key_index_list, seed=i)
                self.log(f"  Shuffled key indices.")

            # Calculate block size k. 
            # Algorithm: k <- f(p). Assuming standard doubling strategy or fixed initial.
            # Using simple doubling strategy: k = initial * 2^(i-1)
            if i == 1:
                k = int(0.73 / qber if 1 > qber > 0  else 9) # Avoid division by zero and extreme qber
            else:
                k = k * (2 ** (i - 1)) #self.initial_block_size * (2 ** (i - 1))
            
            self.log(f"  Block size k: {k}")

            block_index_lists = []
            if i not in self.net_block_index_lists:
                self.net_block_index_lists[i] = []

            # Partition indices into blocks
            num_blocks = math.ceil(self.N / k)
            for l in range(1, num_blocks + 1):
                start_idx = (l - 1) * k
                end_idx = min(l * k, self.N)
                # python slice is exclusive at end, algorithm implies inclusive range logic
                block = key_index_list[start_idx : end_idx]
                block_index_lists.append(block)

            # "Bob sends block_index_lists to Alice" -> Channel Use (Bulk send)
            self.channel_uses += 1

            # "Alice computes Alice_parity_list"
            Alice_parity_list = self.parity(self.P_int, block_index_lists)
            ###############!!!!!!!!!!!!!!!!!!!!!!!!
            self.bits_revealed += len(Alice_parity_list)
            
            # "Alice sends Alice_parity_list to Bob"
            
            Bob_parity_list = self.parity(self.Q_int, block_index_lists)
            
            # Find blocks with odd error parity
            odd_error_parity_block_index_list = [] # stores indices j
            for j in range(len(Alice_parity_list)):
                if Alice_parity_list[j] != Bob_parity_list[j]:
                    odd_error_parity_block_index_list.append(j)
            
            self.log(f"  Blocks with odd parity errors: {len(odd_error_parity_block_index_list)} / {len(block_index_lists)}")

            # Add correct blocks to net_block_index_lists
            # Algorithm: range([0 ... ceil(N/k)-1] \ odd_error...)
            all_block_indices = set(range(len(block_index_lists)))
            correct_block_indices = all_block_indices - set(odd_error_parity_block_index_list)
            
            for j in correct_block_indices:
                self.net_block_index_lists[i].append(block_index_lists[j])

            net_index_list_for_bit_flip = []
            complete_index_list_for_bit_flip = []
            
            # Process bad blocks
            for j in odd_error_parity_block_index_list:
                self.log(f"  Correcting error in block {j}...")
                bit_flip_index = self.binary(block_index_lists[j], i)
                
                # Flip bit in Q_int
                # Q_int <- Q_int XOR (1 << (N - bit_flip_index - 1))
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
                    for m in range(i - 1, 0, -1): # i-1 down to 1
                         if m in self.net_block_index_lists:
                             for sub_list in self.net_block_index_lists[m]:
                                 if bit_flip_idx in sub_list:
                                     if sub_list not in net_check_block_index_lists:
                                         net_check_block_index_lists.append(sub_list)
                
                while len(net_check_block_index_lists) > 0:
                    # check_block_index_list <- min_length(net_check_block_index_lists)
                    # Find smallest block
                    check_block_index_list = min(net_check_block_index_lists, key=len)
                    
                    # check_block_pass_num <- Key which ^ check_block in net_block_index_lists
                    check_block_pass_num = -1
                    for m_key, lists in self.net_block_index_lists.items():
                        if check_block_index_list in lists:
                            check_block_pass_num = m_key
                            break
                    
                    net_check_block_index_lists.remove(check_block_index_list)
                    
                    parity_check = 0
                    
                    # Check how many errors corrected so far are in this block
                    for error_index in complete_index_list_for_bit_flip:
                        if error_index in check_block_index_list:
                            parity_check += 1
                    
                    if parity_check % 2 != 0:
                        self.log(f"    [CASCADE] Re-evaluating block from Pass {check_block_pass_num} (len {len(check_block_index_list)})")
                        
                        # Remove from known good blocks
                        if check_block_pass_num != -1:
                            if check_block_index_list in self.net_block_index_lists[check_block_pass_num]:
                                self.net_block_index_lists[check_block_pass_num].remove(check_block_index_list)
                        
                        # Binary search to find new error
                        new_bit_flip_index = self.binary(check_block_index_list, check_block_pass_num)
                        
                        # Flip bit
                        mask = 1 << (self.N - new_bit_flip_index - 1)
                        self.Q_int = self.Q_int ^ mask
                        self.total_errors_corrected += 1
                        
                        complete_index_list_for_bit_flip.append(new_bit_flip_index)
                        
                        check_block_new_additions = []
                        
                        # Look for other blocks affected by this new flip
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

# Example Usage:
if __name__ == "__main__":
    alice_str = "10110101" * 8
    bob_str   = "10110100" * 8 # Last bit flipped
    
    alice_key = [int(x) for x in alice_str]
    bob_key   = [int(x) for x in bob_str]

    protocol = CascadeProtocol(num_passes=4)
    result_key, bits, errors, channels = protocol.run(alice_key, bob_key, qber=0.125)
    
    print(f"\nFinal Key Match: {result_key == alice_key}")
    print(f"Bits Revealed: {bits}")
    print(f"Errors Corrected: {errors}")
    print(f"Channel Uses: {channels}")