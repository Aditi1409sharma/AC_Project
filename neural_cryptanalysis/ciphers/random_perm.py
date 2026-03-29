import random
from .base import BlockCipher

class RandomPermutation(BlockCipher):
    def __init__(self, block_bits: int):
        super().__init__(block_bits, block_bits)

    def encrypt(self, plaintext: int, key: int, rounds: int) -> int:
        # Convert to Python int and keep within 64-bit range
        seed = (int(key) ^ (int(plaintext) * 6364136223846793005)) & ((1 << 64) - 1)
        rng = random.Random(seed)
        return rng.getrandbits(self.block_bits)