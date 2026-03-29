import numpy as np
from .utils import hamming_weight, int_to_bits


def transform(c: int, c_: int, bits: int) -> np.ndarray:
    """
    Statistical feature vector (7 features, all normalized to [0, 1]):

    1. HW(C)  / bits              — weight of first ciphertext
    2. HW(C') / bits              — weight of second ciphertext
    3. HW(ΔC) / bits              — weight of XOR difference
    4. |HW(C) - HW(C')| / bits   — differential hamming weight
    5. dot(bits(C), bits(C'))/bits — bit-level correlation
    6. HW(upper half of ΔC)/half  — diffusion in upper word
    7. HW(lower half of ΔC)/half  — diffusion in lower word
    """
    diff = c ^ c_
    bc   = int_to_bits(c,    bits).astype(np.float32)
    bc_  = int_to_bits(c_,   bits).astype(np.float32)

    hw_c   = hamming_weight(c)    / bits
    hw_c_  = hamming_weight(c_)   / bits
    hw_d   = hamming_weight(diff) / bits
    hw_xor = abs(hamming_weight(c) - hamming_weight(c_)) / bits
    corr   = float(np.dot(bc, bc_)) / bits

    half   = bits // 2
    hw_hi  = hamming_weight(diff >> half)               / half
    hw_lo  = hamming_weight(diff & ((1 << half) - 1))   / half

    return np.array([hw_c, hw_c_, hw_d, hw_xor, corr, hw_hi, hw_lo],
                    dtype=np.float32)