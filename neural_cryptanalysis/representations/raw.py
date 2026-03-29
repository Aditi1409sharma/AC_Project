import numpy as np
from .utils import int_to_bits

def transform(c, c_, bits):
    """
    Raw representation: bits(C) || bits(C')
    Concatenation of both ciphertexts as bit vectors.
    Shape: (2 * bits,)
    """
    return np.concatenate([
        int_to_bits(c,  bits),
        int_to_bits(c_, bits)
    ]).astype(np.float32)