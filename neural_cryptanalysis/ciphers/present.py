from neural_cryptanalysis.ciphers.base import BlockCipher

class PresentCipher(BlockCipher):
    """PRESENT-80: 64-bit block, 80-bit key, up to 31 rounds."""

    FULL_ROUNDS = 31

    SBOX = [0xC,0x5,0x6,0xB,0x9,0x0,0xA,0xD,
            0x3,0xE,0xF,0x8,0x4,0x7,0x1,0x2]

    def __init__(self):
        super().__init__(block_bits=64, key_bits=80)

    def _sub_layer(self, state):
        out = 0
        for i in range(16):
            out |= self.SBOX[(state >> (4*i)) & 0xF] << (4*i)
        return out

    def _p_layer(self, state):
        out = 0
        for i in range(63):
            if (state >> i) & 1:
                out |= 1 << ((16*i) % 63)
        if (state >> 63) & 1:   # bit 63 stays at 63
            out |= 1 << 63
        return out

    def _key_schedule(self, key, rounds):
        MASK80 = (1 << 80) - 1
        schedule = []
        K = key & MASK80

        for r in range(1, rounds + 2):   # rounds+1 keys needed
            schedule.append(K >> 16)     # top 64 bits = round key

            if r <= rounds:
                # 1. rotate left by 61 (= right by 19)
                K = ((K << 61) | (K >> 19)) & MASK80
                # 2. S-box on top nibble (bits 79:76)
                top4 = (K >> 76) & 0xF
                K = (K & ~(0xF << 76)) | (self.SBOX[top4] << 76)
                # 3. XOR 5-bit round counter into bits 19:15
                K ^= (r & 0x1F) << 15

        return schedule

    def encrypt(self, plaintext, key, rounds):
        assert 1 <= rounds <= self.FULL_ROUNDS
        ks    = self._key_schedule(key, rounds)
        state = plaintext

        for r in range(rounds):
            state ^= ks[r]
            state  = self._sub_layer(state)
            state  = self._p_layer(state)

        state ^= ks[rounds]   # final key whitening
        return state

    def random_key(self):
        import random
        return random.getrandbits(80)