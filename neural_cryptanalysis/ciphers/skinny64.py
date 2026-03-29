from neural_cryptanalysis.ciphers.base import BlockCipher

class Skinny64Cipher(BlockCipher):
    FULL_ROUNDS = 32

    SBOX = [0xC, 0x6, 0x9, 0x0, 0x1, 0xA, 0x2, 0xB,
            0x3, 0x8, 0x5, 0xD, 0x4, 0xE, 0x7, 0xF]

    # Tweakey permutation PT
    PT = [9, 15, 8, 13, 10, 14, 12, 11, 0, 1, 2, 3, 4, 5, 6, 7]

    def __init__(self):
        super().__init__(block_bits=64, key_bits=64)
        self.RC = self._generate_rc()

    def _generate_rc(self):
        rc = []
        c = 0x01
        for _ in range(self.FULL_ROUNDS):
            rc.append(c)
            # feedback = XNOR of bit5 and bit4 (NOT XOR)
            feedback = 1 ^ (((c >> 5) ^ (c >> 4)) & 1)
            c = ((c << 1) & 0x3F) ^ feedback
        return rc

    def _to_nibbles(self, x):
        """Convert 64-bit int to list of 16 nibbles, MSN first."""
        return [(x >> (60 - 4 * i)) & 0xF for i in range(16)]

    def _from_nibbles(self, ns):
        """Convert list of 16 nibbles back to 64-bit int."""
        return sum(ns[i] << (60 - 4 * i) for i in range(16))

    def encrypt(self, plaintext, key, rounds):
        assert 1 <= rounds <= self.FULL_ROUNDS

        s  = self._to_nibbles(plaintext)
        tk = self._to_nibbles(key)

        for r in range(rounds):

            # Step 1: SubCells
            s = [self.SBOX[x] for x in s]

            # Step 2: AddConstants
            c = self.RC[r]
            s[0]  ^= (c & 0xF)         # c0 = bits 3..0 → row 0, col 0
            s[4]  ^= (c >> 4) & 0x3    # c1 = bits 5..4 → row 1, col 0
            s[8]  ^= 0x2               # c2 = 0x2 always → row 2, col 0

            # Step 3: AddRoundTweakey — XOR first two rows with TK
            for i in range(8):
                s[i] ^= tk[i]

            # Step 4: ShiftRows
            # Row 0: no shift
            # Row 1: rotate RIGHT by 1 (last element wraps to front)
            s[4], s[5], s[6], s[7] = s[7], s[4], s[5], s[6]
            # Row 2: rotate right by 2
            s[8], s[9], s[10], s[11] = s[10], s[11], s[8], s[9]
            # Row 3: rotate right by 3 (= rotate left by 1)
            s[12], s[13], s[14], s[15] = s[13], s[14], s[15], s[12]

            # Step 5: MixColumns
            for j in range(4):
                c0, c1, c2, c3 = s[j], s[4+j], s[8+j], s[12+j]
                s[j]    = c0 ^ c2 ^ c3
                s[4+j]  = c0
                s[8+j]  = c1 ^ c2
                s[12+j] = c0 ^ c2

            # Step 6: Update Tweakey for next round (permutation only — no LFSR for TK1)
            tk = [tk[self.PT[i]] for i in range(16)]

        return self._from_nibbles(s)

    def random_key(self):
        import random
        return random.getrandbits(64)