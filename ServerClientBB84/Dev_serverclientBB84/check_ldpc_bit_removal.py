import asyncio
import numpy as np

from nr_ldpc_standard import NR_LDPC_Standard_ClientProtocol


class MockOracle:
    def __init__(self, key):
        self.key = np.array(key, dtype=int)

    async def get_parities(self, blocks):
        return [int(np.sum(self.key[b]) % 2) for b in blocks]


async def main():
    n = 5000
    qber = 0.05

    rng = np.random.default_rng(123)
    alice_key = rng.integers(0, 2, size=n).tolist()
    bob_key = alice_key.copy()

    flip_indices = rng.choice(n, size=int(n * qber), replace=False)
    for idx in flip_indices:
        bob_key[idx] = 1 - bob_key[idx]

    protocol = NR_LDPC_Standard_ClientProtocol(verbose=False, rate=0.333)
    oracle = MockOracle(alice_key)

    corrected_key, bits_revealed, errors, channel_uses = await protocol.run(bob_key, qber, oracle)

    print(f"input key length:    {len(bob_key)}")
    print(f"corrected key length:{len(corrected_key)}")
    print(f"bits revealed:       {bits_revealed}")
    print(f"channel uses:        {channel_uses}")
    print(f"errors reported:     {errors}")

    if len(corrected_key) == len(bob_key):
        print("Result: LDPC reconciliation did not remove bits from the key; it kept the same length.")
    elif len(corrected_key) < len(bob_key):
        print("Result: Some bits were removed during LDPC processing.")
    else:
        print("Result: Output key is longer than input key, which would be unexpected.")


if __name__ == "__main__":
    asyncio.run(main())