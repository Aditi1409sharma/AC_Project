import numpy as np
from .utils import hamming_weight

def transform(c, c_, bits):
    """
    Statistical features:
    - HW(C)
    - HW(C′)
    - HW(C ⊕ C′)
    """
    return np.array([
        hamming_weight(c),
        hamming_weight(c_),
        hamming_weight(c ^ c_)
    ], dtype=np.float32)