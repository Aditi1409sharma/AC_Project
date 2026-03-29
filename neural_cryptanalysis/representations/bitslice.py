import numpy as np
from .utils import int_to_bits
 
 
def transform(c, c_, bits):
    """
    Bit-slice representation.
    Shape = (2, bits) — channel-first, CNN-friendly.
    axis=0 gives 2 channels (C and C'), each of length `bits`.
    """
    b1 = int_to_bits(c,  bits)
    b2 = int_to_bits(c_, bits)
    return np.stack([b1, b2], axis=0)   # shape: (2, bits)