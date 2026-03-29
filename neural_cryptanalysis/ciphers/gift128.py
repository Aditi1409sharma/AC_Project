from neural_cryptanalysis.ciphers.base import BlockCipher


class Gift128Cipher(BlockCipher):
    """
    GIFT-128: 128-bit block, 128-bit key, 40 rounds.
    Follows the GIFT-128 (non-COFB) specification.
    Reference: Banik et al., CHES 2017.
    """

    FULL_ROUNDS = 40

    SBOX = [0x1, 0xA, 0x4, 0xC, 0x6, 0xF, 0x3, 0x9,
            0x2, 0xD, 0xB, 0x7, 0x5, 0x0, 0x8, 0xE]

    # Bit permutation for GIFT-128 (128-entry table)
    PERM = [
          0,  33,  66,  99,  96,   1,  34,  67,  64,  97,   2,  35,  32,  65,  98,   3,
          4,  37,  70, 103, 100,   5,  38,  71,  68, 101,   6,  39,  36,  69, 102,   7,
          8,  41,  74, 107, 104,   9,  42,  75,  72, 105,  10,  43,  40,  73, 106,  11,
         12,  45,  78, 111, 108,  13,  46,  79,  76, 109,  14,  47,  44,  77, 110,  15,
         16,  49,  82, 115, 112,  17,  50,  83,  80, 113,  18,  51,  48,  81, 114,  19,
         20,  53,  86, 119, 116,  21,  54,  87,  84, 117,  22,  55,  52,  85, 118,  23,
         24,  57,  90, 123, 120,  25,  58,  91,  88, 121,  26,  59,  56,  89, 122,  27,
         28,  61,  94, 127, 124,  29,  62,  95,  92, 125,  30,  63,  60,  93, 126,  31,
    ]

    # 6-bit round constants (same sequence as GIFT-64, extended to 40 rounds)
    RC = [
        0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3E, 0x3D, 0x3B, 0x37, 0x2F,
        0x1E, 0x3C, 0x39, 0x33, 0x27, 0x0E, 0x1D, 0x3A, 0x35, 0x2B,
        0x16, 0x2C, 0x18, 0x30, 0x21, 0x02, 0x05, 0x0B, 0x17, 0x2E,
        0x1C, 0x38, 0x31, 0x23, 0x06, 0x0D, 0x1B, 0x36, 0x2D, 0x1A,
    ]

    def __init__(self):
        super().__init__(block_bits=128, key_bits=128)

    @staticmethod
    def _ror16(x, n):
        return ((x >> n) | (x << (16 - n))) & 0xFFFF

    @staticmethod
    def _bitpos(i, total=128):
        return total - 1 - i

    def encrypt(self, plaintext: int, key: int, rounds: int) -> int:
        assert 1 <= rounds <= self.FULL_ROUNDS

        # Key: 8 x 16-bit words
        W = [(key >> (16 * i)) & 0xFFFF for i in range(8)]
        state = plaintext & ((1 << 128) - 1)

        for r in range(rounds):
            # 1. SubCells — nibble-wise S-box
            s = 0
            for i in range(32):
                nibble = (state >> (4 * i)) & 0xF
                s |= self.SBOX[nibble] << (4 * i)
            state = s

            # 2. PermBits
            s = 0
            for i in range(128):
                if (state >> self._bitpos(i)) & 1:
                    s |= 1 << self._bitpos(self.PERM[i])
            state = s

            # 3. AddRoundKey
            U, V = W[1], W[0]
            for i in range(32):
                if (V >> (i % 16)) & 1:
                    state ^= 1 << self._bitpos(4 * i)
                if (U >> (i % 16)) & 1:
                    state ^= 1 << self._bitpos(4 * i + 1)

            # Flip bit 127 (MSB in spec = LSB in Python for bit 127)
            state ^= 1 << self._bitpos(127)

            # Round constants into bits 3, 7, 11, 15, 19, 23
            for j in range(6):
                if (self.RC[r] >> j) & 1:
                    state ^= 1 << self._bitpos(4 * j + 3)

            # 4. Key schedule
            W = [
                W[2], W[3], W[4], W[5],
                W[6], W[7],
                self._ror16(W[0], 12),
                self._ror16(W[1], 2),
            ]

        return state

    def random_key(self) -> int:
        import random
        return random.getrandbits(128)
