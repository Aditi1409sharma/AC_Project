import numpy as np
 
 
def int_to_bits(x: int, bits: int) -> np.ndarray:
    x = int(x)   # ← add this line to handle numpy integers
    n_bytes = (bits + 7) // 8
    b   = x.to_bytes(n_bytes, byteorder='little')
    arr = np.frombuffer(b, dtype=np.uint8)
    return np.unpackbits(arr, bitorder='little')[:bits]
 
 
def hamming_weight(x: int) -> int:
    return bin(x).count("1")