import numpy as np

def int_to_bits(x: int, bits: int) -> np.ndarray:
    """
    Convert integer to bit vector (LSB first).
    """
    return np.array([(x >> i) & 1 for i in range(bits)], dtype=np.uint8)


def hamming_weight(x: int) -> int:
    return bin(x).count("1")