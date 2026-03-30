import numpy as np
from .utils import hamming_weight, int_to_bits


def transform(c: int, c_: int, bits: int) -> np.ndarray:
    """
    Rich statistical feature vector — 3 groups of features:

    GROUP 1 — Global HW stats (7 features):
      1.  HW(C)  / bits
      2.  HW(C') / bits
      3.  HW(ΔC) / bits
      4.  |HW(C) - HW(C')| / bits
      5.  dot(bits(C), bits(C')) / bits   — bit-level correlation
      6.  HW(upper half of ΔC) / half     — diffusion upper word
      7.  HW(lower half of ΔC) / half     — diffusion lower word

    GROUP 2 — Nibble-level HW distribution (bits//2 features):
      For each nibble position i in ΔC:
        HW(nibble_i) / 4                  — per-nibble weight of XOR diff

    GROUP 3 — Bit-position features (bits features):
      XOR bit vector bits(C XOR C')       — raw differential bits
      (same as delta rep, gives positional signal)

    GROUP 4 — Byte-level stats (bits//4 features):
      For each byte position i in ΔC:
        HW(byte_i) / 8                    — per-byte weight of XOR diff

    GROUP 5 — Autocorrelation of ΔC bits (8 lags):
      autocorr(bits(ΔC), lag=k) for k in 1..8

    Total shape: 7 + bits//2 + bits + bits//4 + 8
    For simon32 (bits=32): 7 + 16 + 32 + 8 + 8 = 71 features
    """
    diff  = c ^ c_
    bc    = int_to_bits(c,    bits).astype(np.float32)
    bc_   = int_to_bits(c_,   bits).astype(np.float32)
    bdiff = int_to_bits(diff, bits).astype(np.float32)

    # ── Group 1: global HW stats ──────────────────────────────────────────────
    hw_c   = hamming_weight(c)    / bits
    hw_c_  = hamming_weight(c_)   / bits
    hw_d   = hamming_weight(diff) / bits
    hw_xor = abs(hamming_weight(c) - hamming_weight(c_)) / bits
    corr   = float(np.dot(bc, bc_)) / bits
    half   = bits // 2
    hw_hi  = hamming_weight(diff >> half)              / half
    hw_lo  = hamming_weight(diff & ((1 << half) - 1))  / half

    g1 = np.array([hw_c, hw_c_, hw_d, hw_xor, corr, hw_hi, hw_lo],
                  dtype=np.float32)

    # ── Group 2: per-nibble HW of ΔC ─────────────────────────────────────────
    n_nibbles = bits // 4
    g2 = np.array(
        [bin((diff >> (4 * i)) & 0xF).count('1') / 4.0 for i in range(n_nibbles)],
        dtype=np.float32
    )

    # ── Group 3: raw XOR bit vector (positional differential signal) ──────────
    g3 = bdiff  # shape (bits,)

    # ── Group 4: per-byte HW of ΔC ───────────────────────────────────────────
    n_bytes = bits // 8
    if n_bytes > 0:
        g4 = np.array(
            [bin((diff >> (8 * i)) & 0xFF).count('1') / 8.0 for i in range(n_bytes)],
            dtype=np.float32
        )
    else:
        g4 = np.array([], dtype=np.float32)

    # ── Group 5: autocorrelation of ΔC bits at lags 1..8 ─────────────────────
    lags = min(8, bits - 1)
    mean_d = bdiff.mean()
    var_d  = bdiff.var() + 1e-8
    g5 = np.array(
        [float(np.mean((bdiff[k:] - mean_d) * (bdiff[:-k] - mean_d))) / var_d
         for k in range(1, lags + 1)],
        dtype=np.float32
    )

    return np.concatenate([g1, g2, g3, g4, g5])
