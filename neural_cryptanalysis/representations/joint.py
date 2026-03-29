import numpy as np
from .utils import int_to_bits


def transform(p: int, p_: int, c: int, c_: int, bits: int) -> np.ndarray:
    """
    Extended joint plaintext-ciphertext representation.

    Includes:
    - bits(P)       : plaintext
    - bits(P')      : paired plaintext
    - bits(C)       : ciphertext
    - bits(C')      : paired ciphertext
    - bits(P XOR P'): input difference (fixed delta — constant but useful as anchor)
    - bits(C XOR C'): output difference (the key differential signal)

    Total: 6 * block_bits features.

    Rationale: giving the model both the input and output differences
    explicitly lets it learn the differential propagation pattern directly,
    rather than having to compute it internally.
    """
    p   = int(p)
    p_  = int(p_)
    c   = int(c)
    c_  = int(c_)

    return np.concatenate([
        int_to_bits(p,       bits),   # plaintext
        int_to_bits(p_,      bits),   # paired plaintext
        int_to_bits(c,       bits),   # ciphertext
        int_to_bits(c_,      bits),   # paired ciphertext
        int_to_bits(p ^ p_,  bits),   # input XOR difference
        int_to_bits(c ^ c_,  bits),   # output XOR difference
    ])
