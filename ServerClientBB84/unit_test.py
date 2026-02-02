import sys
import os
import asyncio
import random
import math

# Add directories to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ErrorCorrection.cascade import CascadeProtocol as OriginalCascade
from ServerClientBB84.cascade_refactored import CascadeClientProtocol as RefactoredCascade
from ServerClientBB84.prng import PRNG

class MockAliceOracle:
    """
    Implements the ParityOracle protocol using Alice's key (P_key).
    Used to test the RefactoredCascade protocol against the Original version.
    """
    def __init__(self, p_key):
        self.p_key = p_key
        self.N = len(p_key)
        # Convert key to int using same logic as OriginalCascade
        p_str = "".join(map(str, p_key))
        self.p_int = int(p_str, 2)

    def calculate_parity(self, indices):
        p_val = 0
        for idx in indices:
            shift = self.N - 1 - idx
            bit = (self.p_int >> shift) & 1
            p_val ^= bit
        return p_val

    async def get_parity(self, indices):
        return self.calculate_parity(indices)

    async def get_parities(self, blocks):
        return [self.calculate_parity(block) for block in blocks]

def generate_keys(n, qber, prng: PRNG):
    """
    Generate Alice's key and Bob's key with a certain QBER.
    """
    alice_key = [prng.get_bit() for _ in range(n)]
    bob_key = list(alice_key)
    
    # Introduce errors
    num_errors = int(n * qber)
    indices = prng.sample(range(n), num_errors)
    for idx in indices:
        bob_key[idx] = 1 - bob_key[idx]
        
    return alice_key, bob_key

async def test_consistency(n=1024, passes=4, qber=0.05, seed=42):
    random.seed(seed)
    prng = PRNG(seed)
    alice_key, bob_key = generate_keys(n, qber, prng)
    
    initial_errors = sum(1 for a, b in zip(alice_key, bob_key) if a != b)
    
    print(f"Testing Consistency with N={n}, Passes={passes}, QBER={qber}")
    print(f"Initial Errors: {initial_errors}")
    print("-" * 50)

    # 1. Run Original Cascade
    orig_protocol = OriginalCascade(num_passes=passes, verbose=False)
    orig_res_key, orig_bits, orig_errors, orig_channels = orig_protocol.run(alice_key, bob_key, qber)
    
    # 2. Run Refactored Cascade
    ref_protocol = RefactoredCascade(num_passes=passes, verbose=False)
    oracle = MockAliceOracle(alice_key)
    ref_res_key, ref_bits, ref_errors, ref_channels = await ref_protocol.run(bob_key, qber, oracle)

    # Comparisons
    key_match = (orig_res_key == ref_res_key)
    bits_match = (orig_bits == ref_bits)
    errors_match = (orig_errors == ref_errors)
    channels_match = (orig_channels == ref_channels)
    
    success = key_match and bits_match and errors_match and channels_match

    print(f"Keys Match:             {key_match}")
    print(f"Bits Revealed Match:    {bits_match} (Orig: {orig_bits}, Ref: {ref_bits})")
    print(f"Errors Corrected Match: {errors_match} (Orig: {orig_errors}, Ref: {ref_errors})")
    print(f"Channel Uses Match:     {channels_match} (Orig: {orig_channels}, Ref: {ref_channels})")
    
    # Check if keys are actually corrected (Alice and Bob should match)
    final_success = (orig_res_key == alice_key)
    print(f"Original Successfully Corrected: {final_success}")
    print(f"Refactored Successfully Corrected: {ref_res_key == alice_key}")
    
    if not success:
         print("\nMISMATCH DETECTED!")
    else:
         print("\nSUCCESS: Both implementations are mathematically equivalent.")
    
    return success

if __name__ == "__main__":
    # Test with a few different parameters
    asyncio.run(test_consistency(n=1024, qber=0.01))
    print("\n" + "="*50 + "\n")
    asyncio.run(test_consistency(n=1024, qber=0.05))
    print("\n" + "="*50 + "\n")
    asyncio.run(test_consistency(n=2048, qber=0.08))
