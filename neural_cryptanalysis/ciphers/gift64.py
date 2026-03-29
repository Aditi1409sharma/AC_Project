from neural_cryptanalysis.ciphers.base import BlockCipher

class Gift64Cipher(BlockCipher):
    FULL_ROUNDS = 28

    SBOX = [0x1,0xA,0x4,0xC,0x6,0xF,0x3,0x9,
            0x2,0xD,0xB,0x7,0x5,0x0,0x8,0xE]

    PERM = [
         0,17,34,51,48, 1,18,35,32,49, 2,19,16,33,50, 3,
         4,21,38,55,52, 5,22,39,36,53, 6,23,20,37,54, 7,
         8,25,42,59,56, 9,26,43,40,57,10,27,24,41,58,11,
        12,29,46,63,60,13,30,47,44,61,14,31,28,45,62,15
    ]

    RC = [0x01,0x03,0x07,0x0F,0x1F,0x3E,0x3D,0x3B,0x37,0x2F,
          0x1E,0x3C,0x39,0x33,0x27,0x0E,0x1D,0x3A,0x35,0x2B,
          0x16,0x2C,0x18,0x30,0x21,0x02,0x05,0x0B]

    def __init__(self):
        super().__init__(block_bits=64, key_bits=128)

    @staticmethod
    def _ror16(x, n):
        return ((x >> n) | (x << (16 - n))) & 0xFFFF

    @staticmethod
    def _bitpos(i):
        """Convert spec bit index → Python bit index"""
        return 63 - i

    def encrypt(self, plaintext, key, rounds):
        assert 1 <= rounds <= self.FULL_ROUNDS

        # Split key into 8 x 16-bit words
        W = [(key >> (16 * i)) & 0xFFFF for i in range(8)]

        state = plaintext

        for r in range(rounds):

            # 1. SubCells (nibble-wise, safe)
            s = 0
            for i in range(16):
                nibble = (state >> (4 * i)) & 0xF
                s |= self.SBOX[nibble] << (4 * i)
            state = s

            # 2. PermBits (fix indexing)
            s = 0
            for i in range(64):
                if (state >> self._bitpos(i)) & 1:
                    s |= 1 << self._bitpos(self.PERM[i])
            state = s

            # 3. AddRoundKey
            U, V = W[1], W[0]

            for i in range(16):
                if (V >> i) & 1:
                    state ^= 1 << self._bitpos(4 * i)
                if (U >> i) & 1:
                    state ^= 1 << self._bitpos(4 * i + 1)

            # Anti-complementarity: flip b63 (LSB in spec)
            state ^= 1 << self._bitpos(63)

            # Round constants → bits 3,7,...,23
            for j in range(6):
                if (self.RC[r] >> j) & 1:
                    state ^= 1 << self._bitpos(4 * j + 3)

            # 4. Key schedule
            W = [
                W[2], W[3], W[4], W[5],
                W[6], W[7],
                self._ror16(W[0], 12),
                self._ror16(W[1], 2)
            ]

        return state

    def random_key(self):
        import random
        return random.getrandbits(128)