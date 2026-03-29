# ciphers/simon.py
from neural_cryptanalysis.ciphers.base import BlockCipher

class SimonCipher(BlockCipher):
    """
    Simon32/64 — 32-bit block, 64-bit key.
    Supports reduced-round encryption for ML distinguisher experiments.
    """

    # z0 sequence from the original Simon/Speck paper (NSA 2013)
    z0 = 0b01100111000011010100100010111110110011100001101010010001011111

    FULL_ROUNDS = 32
    WORD_SIZE   = 16
    MOD_MASK    = (1 << 16) - 1   # 0xFFFF

    def __init__(self):
        super().__init__(block_bits=32, key_bits=64)

    def _key_schedule(self, key: int) -> list[int]:
        W = self.WORD_SIZE
        M = self.MOD_MASK

        # Split 64-bit key into 4 x 16-bit words
        # k[0] = least significant word, k[3] = most significant word
        words = []
        k = key
        for _ in range(4):
            words.append(k & M)
            k >>= W
        # words = [k0, k1, k2, k3]

        schedule = list(words)   # will grow to FULL_ROUNDS entries

        for i in range(self.FULL_ROUNDS - 4):
            ki3 = schedule[i + 3]   # k[i+3]
            ki1 = schedule[i + 1]   # k[i+1]
            ki  = schedule[i]       # k[i]

            # tmp = S^{-3}(k[i+3])  →  rotate right by 3
            tmp = ((ki3 >> 3) | (ki3 << (W - 3))) & M
            # XOR with k[i+1]  (m=4 case)
            tmp = tmp ^ ki1
            # XOR with S^{-1}(tmp)  →  rotate right by 1
            tmp = tmp ^ (((tmp >> 1) | (tmp << (W - 1))) & M)

            z_bit = (self.z0 >> (i % 62)) & 1
            new_k = (~ki & M) ^ tmp ^ z_bit ^ 3
            schedule.append(new_k)

        return schedule   # length = FULL_ROUNDS = 32

    def encrypt(self, plaintext: int, key: int, rounds: int) -> int:
        assert 1 <= rounds <= self.FULL_ROUNDS, \
            f"rounds must be 1–{self.FULL_ROUNDS}, got {rounds}"

        W = self.WORD_SIZE
        M = self.MOD_MASK

        schedule = self._key_schedule(key)

        # Split 32-bit block into two 16-bit words
        x = (plaintext >> W) & M   # upper word
        y = plaintext & M          # lower word

        for i in range(rounds):
            k   = schedule[i]
            ls1 = ((x << 1) | (x >> (W - 1))) & M   # rotate left 1
            ls8 = ((x << 8) | (x >> (W - 8))) & M   # rotate left 8
            ls2 = ((x << 2) | (x >> (W - 2))) & M   # rotate left 2
            new_y = y ^ (ls1 & ls8) ^ ls2 ^ k
            y = x
            x = new_y

        return (x << W) | y

    def random_key(self) -> int:
        import random
        return random.getrandbits(self.key_bits)