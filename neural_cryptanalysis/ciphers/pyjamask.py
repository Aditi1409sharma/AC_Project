from neural_cryptanalysis.ciphers.base import BlockCipher


class PyjamaskCipher(BlockCipher):
    """
    Pyjamask-96: 96-bit block, 128-bit key, 14 rounds.
    Reference: Goudarzi et al., CHES 2020 / ToSC 2019.

    State: three 32-bit rows (r0, r1, r2).
    Key schedule: 128-bit key expanded into 15 round keys (each 96-bit = 3 x 32-bit).
    """

    FULL_ROUNDS = 14

    # S-box: 4-bit → 4-bit (applied to each nibble of each row)
    # Pyjamask uses a 4-bit S-box applied column-wise across the 3 rows
    # Each "column" is 3 bits (one from each row), but the spec uses a
    # column-wise operation. We implement the full column mixing.

    # MixRows constants (one per row) — 32-bit linear feedback
    MR = [0xA3861085, 0x63417021, 0x692CF2F9]

    # Round constants (14 values, 32-bit each)
    RC = [
        0x00000000, 0x13198A2E, 0x03707344, 0x243F6A88,
        0x85A308D3, 0x13198A2E, 0x03707344, 0x243F6A88,
        0x85A308D3, 0x13198A2E, 0x03707344, 0x243F6A88,
        0x85A308D3, 0x13198A2E,
    ]

    def __init__(self):
        super().__init__(block_bits=96, key_bits=128)

    @staticmethod
    def _ror32(x: int, n: int) -> int:
        return ((x >> n) | (x << (32 - n))) & 0xFFFFFFFF

    def _mix_rows(self, r0: int, r1: int, r2: int):
        """
        MixRows: each row is multiplied by its companion matrix.
        Implemented as a series of rotations and XORs.
        """
        def mix_one(row: int, m: int) -> int:
            out = 0
            for i in range(32):
                if (m >> i) & 1:
                    out ^= self._ror32(row, i)
            return out & 0xFFFFFFFF

        return mix_one(r0, self.MR[0]), mix_one(r1, self.MR[1]), mix_one(r2, self.MR[2])

    def _sub_bytes(self, r0: int, r1: int, r2: int):
        """
        SubBytes: apply 3-bit S-box column-wise.
        Each column is (bit_i of r0, bit_i of r1, bit_i of r2).
        S-box: (a,b,c) → (a^bc, a^b^ac, a^b^c^ab)
        This is the Pyjamask non-linear layer.
        """
        # Vectorised over all 32 columns simultaneously using bitwise ops
        a, b, c = r0, r1, r2
        r0_new = a ^ (b & c)
        r1_new = a ^ b ^ (a & c)
        r2_new = a ^ b ^ c ^ (a & b)
        return r0_new & 0xFFFFFFFF, r1_new & 0xFFFFFFFF, r2_new & 0xFFFFFFFF

    def _key_schedule(self, key: int) -> list:
        """
        Expand 128-bit key into (FULL_ROUNDS + 1) x 96-bit round keys.
        Key state: 4 x 32-bit words (k0, k1, k2, k3).
        Each step: mix rows on (k0,k1,k2), then XOR round constant into k3.
        Round key = (k0, k1, k2).
        """
        MASK = 0xFFFFFFFF
        k = [(key >> (96 - 32 * i)) & MASK for i in range(4)]  # k0..k3

        schedule = []
        for r in range(self.FULL_ROUNDS + 1):
            schedule.append((k[0], k[1], k[2]))

            if r < self.FULL_ROUNDS:
                # MixRows on first three words
                k[0], k[1], k[2] = self._mix_rows(k[0], k[1], k[2])
                # XOR round constant into k3
                k[3] ^= self.RC[r]
                # Rotate k3 right by 1 and XOR into k0
                k[0] ^= self._ror32(k[3], 1)

        return schedule

    def encrypt(self, plaintext: int, key: int, rounds: int) -> int:
        assert 1 <= rounds <= self.FULL_ROUNDS

        MASK = 0xFFFFFFFF
        ks = self._key_schedule(key)

        # Split 96-bit plaintext into 3 x 32-bit rows
        r0 = (plaintext >> 64) & MASK
        r1 = (plaintext >> 32) & MASK
        r2 = plaintext & MASK

        for r in range(rounds):
            # AddRoundKey
            rk0, rk1, rk2 = ks[r]
            r0 ^= rk0
            r1 ^= rk1
            r2 ^= rk2

            # SubBytes
            r0, r1, r2 = self._sub_bytes(r0, r1, r2)

            # MixRows
            r0, r1, r2 = self._mix_rows(r0, r1, r2)

        # Final key whitening
        rk0, rk1, rk2 = ks[rounds]
        r0 ^= rk0
        r1 ^= rk1
        r2 ^= rk2

        return (r0 << 64) | (r1 << 32) | r2

    def random_key(self) -> int:
        import random
        return random.getrandbits(128)
