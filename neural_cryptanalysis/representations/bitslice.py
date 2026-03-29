import numpy as np
from .utils import int_to_bits

def transform(c, c_, bits):
    """
    Bit-slice representation:
    shape = (bits, 2)
    """
    b1 = int_to_bits(c, bits)
    b2 = int_to_bits(c_, bits)
    return np.stack([b1, b2], axis=1)