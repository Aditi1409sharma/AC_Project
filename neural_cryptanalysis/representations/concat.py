import numpy as np
from .utils import int_to_bits

def transform(c, c_, bits):
    """
    Concat representation: bits(C) || bits(C′)
    """
    return np.concatenate([
        int_to_bits(c, bits),
        int_to_bits(c_, bits)
    ])