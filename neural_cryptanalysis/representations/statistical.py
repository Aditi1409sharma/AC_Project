import numpy as np
from .utils import hamming_weight, int_to_bits


def transform(c: int, c_: int, bits: int) -> np.ndarray:
    """
    Extended statistical feature vector.

    Features extracted:
    1.  HW(C)                        - Hamming weight of C
    2.  HW(C')                       - Hamming weight of C'
    3.  HW(C XOR C')                 - Hamming weight of difference
    4.  HW(C) / bits                 - Normalised HW(C)
    5.  HW(C') / bits                - Normalised HW(C')
    6.  HW(C XOR C') / bits          - Normalised HW of difference
    7.  |HW(C) - HW(C')|             - Absolute HW imbalance
    8.  HW(C AND C')                 - Bit overlap count
    9.  HW(C OR C')                  - Union bit count
    10. HW(C AND ~C')                - Bits in C but not C'
    11. HW(~C AND C')                - Bits in C' but not C
    12. HW(C XOR C') / HW(C OR C')  - Jaccard-like distance (0 if OR=0)
    13-44. Per-nibble HW of XOR difference (8 nibbles for 32-bit, 16 for 64-bit)
           Captures which nibble positions carry the most differential weight
    """
    mask = (1 << bits) - 1
    c  = int(c)  & mask
    c_ = int(c_) & mask

    xor   = c ^ c_
    and_  = c & c_
    or_   = c | c_
    c_not = (~c) & mask
    c_not_ = (~c_) & mask

    hw_c   = hamming_weight(c)
    hw_cp  = hamming_weight(c_)
    hw_xor = hamming_weight(xor)
    hw_or  = hamming_weight(or_)

    jaccard = hw_xor / hw_or if hw_or > 0 else 0.0

    base_feats = np.array([
        hw_c,
        hw_cp,
        hw_xor,
        hw_c  / bits,
        hw_cp / bits,
        hw_xor / bits,
        abs(hw_c - hw_cp),
        hamming_weight(and_),
        hw_or,
        hamming_weight(c & c_not_),
        hamming_weight(c_not & c_),
        jaccard,
    ], dtype=np.float32)

    # Per-nibble HW of XOR difference
    n_nibbles = bits // 4
    nibble_hws = np.array([
        bin((xor >> (4 * i)) & 0xF).count('1')
        for i in range(n_nibbles)
    ], dtype=np.float32)

    return np.concatenate([base_feats, nibble_hws])
