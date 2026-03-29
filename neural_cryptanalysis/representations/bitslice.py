import numpy as np
from .utils import int_to_bits

def transform(c, c_, bits):
    """
    Bit-slice representation: flattened to 1D for MLP/CNN compatibility.
    Each bit position i has two features: bit_i(C) and bit_i(C').
    Output shape: (bits * 2,) — interleaved [C_bit0, C'_bit0, C_bit1, C'_bit1, ...]

    This lets CNN kernels slide over bit positions and detect patterns
    across corresponding bit positions of C and C' simultaneously.
    """
    b1 = int_to_bits(c, bits)
    b2 = int_to_bits(c_, bits)
    # Interleave: [c0,c'0, c1,c'1, ..., cn,c'n]
    return np.stack([b1, b2], axis=1).flatten()