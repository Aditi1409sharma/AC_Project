import numpy as np
from .utils import int_to_bits

def transform(p, p_, c, c_, bits):
    """
    Joint representation:
    bits(P) || bits(P′) || bits(C) || bits(C′)
    """
    return np.concatenate([
        int_to_bits(p, bits),
        int_to_bits(p_, bits),
        int_to_bits(c, bits),
        int_to_bits(c_, bits),
    ])