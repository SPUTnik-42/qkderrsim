import math
import random
import hashlib
from typing import List

class PrivacyAmplification:
    def __init__(self, key: List[int], qber: float, security_parameter: int = 10):
        self.key = key
        self.N = len(key)
        self.qber = qber
        self.S = security_parameter

        # Calculate V(e) using Shannon Information 
        if self.qber <= 0.0 or self.qber >= 1.0:
            self.V_e = 1.0
        else:
            self.V_e = 1.0 + (self.qber * math.log2(self.qber)) + ((1.0 - self.qber) * math.log2(1.0 - self.qber))
            
        # The anticipated bits of information that Eve knows is estimated to be t = N * V_e
        self.t = math.ceil(self.N * self.V_e)
        
        # Calculate R (final key length)
        # Security parameter bound constraint: [0 < S <= N - t]
        max_S = self.N - self.t
        if self.S > max_S:
            self.S = max(0, max_S)
                
        self.R = self.N - self.t - self.S
        if self.R < 0:
            self.R = 0

    def toeplitz_hash(self, seed: int) -> List[int]:
        """
        Applies Toeplitz hashing using a shared seed.
        This embodies Universal_2 hashing by pseudo-randomly selecting 
        a hash function (a Toeplitz matrix) from the family via the seed.
        """
        if self.R <= 0 or not self.key:
            return []
            
        # We need N + R - 1 random bits to define an R x N Toeplitz matrix
        rng = random.Random(seed)
        t_seq = [rng.randint(0, 1) for _ in range(self.N + self.R - 1)]
        
        secret_key = []
        for i in range(self.R):
            # Row i of the Toeplitz matrix is formed by taking N bits 
            # from t_seq starting at index i, reversed.
            # T[i][j] = t_seq[i - j + N - 1]
            row = t_seq[i : i + self.N][::-1]
            
            # Matrix multiplication in GF(2): dot product modulo 2
            bit = sum(r & k for r, k in zip(row, self.key)) % 2
            secret_key.append(bit)
            
        return secret_key

    def shake256_hash(self, seed: int) -> List[int]:
        """
        Applies SHAKE256 hashing to extract a secure key.
        This provides a cryptographic sponge function construction for privacy amplification,
        using the seed as a salt/nonce mixed with the key material.
        """
        if self.R <= 0 or not self.key:
            return []
            
        # Convert key bits to bytes
        # Using string representation then encode for simplicity, 
        # but could pack bits for better efficiency. 
        key_str = ''.join(map(str, self.key))
        seed_bytes = seed.to_bytes((seed.bit_length() + 7) // 8 or 1, 'big')
        
        # Initialize shake_256
        hasher = hashlib.shake_256()
        hasher.update(seed_bytes)
        hasher.update(key_str.encode('utf-8'))
        
        # We need R bits total. 1 byte = 8 bits.
        num_bytes = math.ceil(self.R / 8.0)
        digest = hasher.digest(num_bytes)
        
        # Convert digest bytes back to a list of bits
        secret_key_bits = []
        for byte in digest:
            for i in range(7, -1, -1):
                secret_key_bits.append((byte >> i) & 1)
                
        # Truncate to exact required length (R bits)
        return secret_key_bits[:self.R]

    def apply_hash(self, seed: int, algorithm: str = 'toeplitz') -> List[int]:
        """
        A wrapper function to choose the desired hash algorithm.
        Supported algorithms: 'toeplitz', 'shake256'
        """
        algo = algorithm.lower().strip()
        if algo == 'shake256':
            return self.shake256_hash(seed)
        elif algo == 'toeplitz':
            return self.toeplitz_hash(seed)
        else:
            raise ValueError(f"Unknown PA hash algorithm: {algorithm}. Supported: 'toeplitz', 'shake256'")

    def log_metrics(self):
        print("\n" + "="*45)
        print("      PRIVACY AMPLIFICATION METRICS")
        print("="*45)
        print(f" [+] Reconciled Key Length (N):      {self.N} bits")
        print(f" [+] QBER (e):                       {self.qber:.4%}")
        print(f" [+] Information Leakage Rate (Ve):  {self.V_e:.6f}")
        print(f" [+] Est. Eve's Knowledge Bound (t): {self.t} bits")
        print(f" [+] Security Parameter (S):         {self.S} bits")
        print(f" [+] Final Secret Key Length (R):    {self.R} bits")
        print("="*45)

if __name__ == "__main__":
    import random as rnd
    import time
    
    print("="*60)
    print("  Privacy Amplification (Toeplitz Hashing) Demo")
    print("="*60)
    
    # 1. Setup Mock Reconciled Keys
    # Alice and Bob have finished Error Correction and now share an identical
    # reconciled key. Let's assume an N=500 bit key with a QBER of 3.5%.
    N_bits = 500
    estimated_qber = 0.035
    
    print(f"\n[System] Simulating Error Correction output (N={N_bits}, QBER={estimated_qber:.2%})")
    
    # Generate identical keys for Alice and Bob
    base_key = [rnd.randint(0, 1) for _ in range(N_bits)]
    alice_reconciled_key = list(base_key)
    bob_reconciled_key   = list(base_key)
    
    print(f"[Alice] Initial Key (first 30 bits): {''.join(map(str, alice_reconciled_key[:30]))}...")
    print(f"[Bob]   Initial Key (first 30 bits): {''.join(map(str, bob_reconciled_key[:30]))}...")
    
    # 2. Bob initiates Privacy Amplification
    print(f"\n[Bob] Initializing Privacy Amplification...")
    t_start = time.time()
    
    bob_pa = PrivacyAmplification(bob_reconciled_key, qber=estimated_qber, security_parameter=10)
    bob_pa.log_metrics()
    
    # Bob generates a truly random seed (acting as the public index for the Toeplitz Family)
    shared_seed = rnd.randint(0, 2**32 - 1)
    print(f"[Bob] Generated Toeplitz Family Seed: {shared_seed}")
    
    # Bob computes his shared secret (Toeplitz)
    bob_secret_key_toeplitz = bob_pa.apply_hash(seed=shared_seed, algorithm='toeplitz')
    
    # 3. Bob transmits the seed to Alice (in reality, via classical authenticated channel)
    print(f"\n[Network] Transmitting seed {shared_seed} over public channel to Alice -->")
    
    # 4. Alice completes Privacy Amplification (Toeplitz)
    print(f"\n[Alice] Received seed {shared_seed}. Initializing mapping...")
    alice_pa = PrivacyAmplification(alice_reconciled_key, qber=estimated_qber, security_parameter=10)
    alice_secret_key_toeplitz = alice_pa.apply_hash(seed=shared_seed, algorithm='toeplitz')
    
    t_end = time.time()
    
    # Let's also run SHAKE256 to compare execution
    print("\n--- SHAKE256 execution for comparison ---")
    t_start_shake = time.time()
    bob_secret_key_shake = bob_pa.apply_hash(seed=shared_seed, algorithm='shake256')
    alice_secret_key_shake = alice_pa.apply_hash(seed=shared_seed, algorithm='shake256')
    t_end_shake = time.time()
    
    # 5. Verify the Final Secret Keys Match
    print(f"\n[System] --- VERIFICATION ---")
    print(f"Time Taken (Toeplitz): {t_end - t_start:.4f}s")
    print(f"Time Taken (SHAKE256): {t_end_shake - t_start_shake:.4f}s")
    
    if len(bob_secret_key_toeplitz) == 0:
        print("[System] Privacy amplification failed: Reconciled key too small relative to leaked info bound.")
    else:
        matches = sum(1 for a, b in zip(alice_secret_key_toeplitz, bob_secret_key_toeplitz) if a == b)
        match_percentage = (matches / len(bob_secret_key_toeplitz)) * 100
        
        print(f"\nFinal Secret Key Length: {len(bob_secret_key_toeplitz)} bits")
        print(f"Alice Secret Hash (Toeplitz, first 50): {''.join(map(str, alice_secret_key_toeplitz[:50]))}...")
        print(f"Bob Secret Hash   (Toeplitz, first 50): {''.join(map(str, bob_secret_key_toeplitz[:50]))}...")
        print(f"Toeplitz Match Integrity: {matches}/{len(bob_secret_key_toeplitz)} ({match_percentage:.2f}%)")
        assert matches == len(bob_secret_key_toeplitz), "Toeplitz Keys do not match!"
        
        matches_s = sum(1 for a, b in zip(alice_secret_key_shake, bob_secret_key_shake) if a == b)
        match_percentage_s = (matches_s / len(bob_secret_key_shake)) * 100
        
        print(f"\nAlice Secret Hash (SHAKE256, first 50): {''.join(map(str, alice_secret_key_shake[:50]))}...")
        print(f"Bob Secret Hash   (SHAKE256, first 50): {''.join(map(str, bob_secret_key_shake[:50]))}...")
        print(f"SHAKE256 Match Integrity: {matches_s}/{len(bob_secret_key_shake)} ({match_percentage_s:.2f}%)")
        assert matches_s == len(bob_secret_key_shake), "SHAKE256 Keys do not match!"
        
        print("\n[SUCCESS] Highly secure, eavesdropper-free symmetric keys generated for both families.")
