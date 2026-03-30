import numpy as np
from .utils import int_to_bits


def transform(c: int, c_: int, bits: int) -> np.ndarray:
    """
    Word-level representation.

    For word-oriented ciphers, splits each ciphertext into 16-bit words
    and normalises each word to [0, 1] by dividing by 0xFFFF.

    This preserves the word-level structure of the cipher (e.g. SIMON
    operates on two 16-bit words) and gives the model a coarser but
    more semantically meaningful view than raw bits.

    Layout: [C_w0, C_w1, ..., C_wN, C'_w0, C'_w1, ..., C'_wN, delta_w0, ...]
    where N = bits // 16.

    For ciphers with block_bits not divisible by 16, falls back to 8-bit
    words (bytes). For very small blocks (< 16 bits), uses 4-bit nibbles.

    Shape: 3 * (bits // word_size)  — C words + C' words + XOR words
    """
    if bits >= 16 and bits % 16 == 0:
        word_size = 16
    elif bits >= 8 and bits % 8 == 0:
        word_size = 8
    else:
        word_size = 4

    mask     = (1 << word_size) - 1
    n_words  = bits // word_size
    norm     = float(mask)

    c_words  = [(c  >> (word_size * i)) & mask for i in range(n_words)]
    cp_words = [(c_ >> (word_size * i)) & mask for i in range(n_words)]
    xor      = c ^ c_
    dx_words = [(xor >> (word_size * i)) & mask for i in range(n_words)]

    vec = np.array(
        [w / norm for w in c_words] +
        [w / norm for w in cp_words] +
        [w / norm for w in dx_words],
        dtype=np.float32
    )
    return vec
