from abc import ABC, abstractmethod

class BlockCipher(ABC):
    def __init__(self, block_bits: int, key_bits: int):
        self.block_bits = block_bits
        self.key_bits   = key_bits

    @abstractmethod
    def encrypt(self, plaintext: int, key: int, rounds: int) -> int:
        """Encrypt a single block. All values are Python ints."""
        ...

    def random_key(self) -> int:
        import random
        return random.getrandbits(self.key_bits)
