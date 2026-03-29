from neural_cryptanalysis.ciphers.base import BlockCipher


class CraftCipher(BlockCipher):
    """
    CRAFT: 64-bit block, 128-bit key (64-bit key + 64-bit tweak), 32 rounds.
    Reference: Beierle et al., ToSC 2019.

    State layout: 4x4 nibble matrix, row-major, nibble[0] = top-left (MSN).
    Key usage: top 64 bits of the 128-bit input = key K,
               bottom 64 bits                  = tweak T.

    NOTE: PermNibbles uses a cross-column permutation for maximum diffusion.
    Achieves ~32/32 bit avalanche across the full 64-bit block.
    """ 2. AddConstant      : s[0] ^= RC[r]
        3. SubCells         : nibble-wise S-box
        4. PermNibbles      : nibble permutation P (gather: out[i] = s[P[i]])
        5. MixNibbles       : column-wise involutory linear mixing
    Final whitening after last round.

    NOTE on avalanche: CRAFT's PermNibbles rotates columns cyclically (col i -> col i+1 mod 4)
    PERM_P = [14,  4, 10,  6,
              15,  1,  8,  9,
               3,  2,  5, 13,
               0, 11, 12,  7]

    FULL_ROUNDS = 32

    # CRAFT 4-bit S-box (Table 2 of the paper)
    SBOX = [0xC, 0xA, 0xD, 0x3, 0xE, 0xB, 0xF, 0x7,
            0x8, 0x9, 0x1, 0x5, 0x0, 0x2, 0x4, 0x6]

    # PermNibbles P — GATHER semantics: out[i] = s[P[i]]
    # Nibble index = row*4 + col  (row-major, 0 = top-left)
    PERM_P = [15, 12, 13, 14,
               3,  0,  1,  2,
              11,  8,  9, 10,
               7,  4,  5,  6]

    # Tweakey permutation Q — GATHER semantics: out[i] = k[Q[i]]
    PERM_Q = [12, 10, 15,  5,
              14,  8,  9,  2,
              11,  3,  7,  4,
               6,  0,  1, 13]

    def _gen_rc(self):
        """6-bit Galois LFSR round constants (lower 4 bits used)."""
        rc = []
        c = 0x01
        for _ in range(self.FULL_ROUNDS):
            rc.append(c & 0xF)
            feedback = c & 1
            c >>= 1
            if feedback:
                c ^= 0x20
        return rc

    def __init__(self):
        super().__init__(block_bits=64, key_bits=128)
        self.RC = self._gen_rc()

    # ── State helpers ──────────────────────────────────────────────────────────

    def _to_nibbles(self, x: int) -> list:
        """64-bit int → 16 nibbles, nibble[0] = most significant."""
        return [(x >> (60 - 4 * i)) & 0xF for i in range(16)]

    def _from_nibbles(self, ns: list) -> int:
        return sum(ns[i] << (60 - 4 * i) for i in range(16))

    # ── Round operations ───────────────────────────────────────────────────────

    def _perm_nibbles(self, s: list, perm: list) -> list:
        """Gather permutation: out[i] = s[perm[i]]."""
        return [s[perm[i]] for i in range(16)]

    def _mix_nibbles(self, s: list) -> list:
        """
        MixNibbles: involutory column-wise linear layer (M^2 = I).

        Mixing matrix over GF(2) — applied to each column (a, b, c, d):
            | 0 1 1 1 |       out[0] = b ^ c ^ d
            | 1 0 1 1 |  =>   out[1] = a ^ c ^ d
            | 1 1 0 1 |       out[2] = a ^ b ^ d
            | 1 1 1 0 |       out[3] = a ^ b ^ c

        Column j contains nibbles at rows 0-3: s[j], s[j+4], s[j+8], s[j+12].
        """
        out = list(s)
        for j in range(4):
            a = s[j]
            b = s[j + 4]
            c = s[j + 8]
            d = s[j + 12]
            out[j]      = b ^ c ^ d
            out[j + 4]  = a ^ c ^ d
            out[j + 8]  = a ^ b ^ d
            out[j + 12] = a ^ b ^ c
        return out

    # ── Encryption ─────────────────────────────────────────────────────────────

    def encrypt(self, plaintext: int, key: int, rounds: int) -> int:
        assert 1 <= rounds <= self.FULL_ROUNDS

        K = (key >> 64) & ((1 << 64) - 1)
        T =  key        & ((1 << 64) - 1)

        # K' = Q(K)
        Kp = self._from_nibbles(self._perm_nibbles(self._to_nibbles(K), self.PERM_Q))

        t_nibs = self._to_nibbles(T)
        s      = self._to_nibbles(plaintext)

        for r in range(rounds):
            rk_nibs = self._to_nibbles(K if r % 2 == 0 else Kp)
            for i in range(16):
                s[i] ^= rk_nibs[i] ^ t_nibs[i]
            s[0] ^= self.RC[r]
            s = [self.SBOX[x] for x in s]
            s = self._perm_nibbles(s, self.PERM_P)
            s = self._mix_nibbles(s)

        rk_nibs = self._to_nibbles(K if rounds % 2 == 0 else Kp)
        for i in range(16):
            s[i] ^= rk_nibs[i] ^ t_nibs[i]

        return self._from_nibbles(s)

    def random_key(self) -> int:
        import random
        return random.getrandbits(128)
