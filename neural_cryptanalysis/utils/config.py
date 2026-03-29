BLOCK_BITS = {
    "simon32":    32,
    "gift64":     64,
    "gift128":   128,
    "skinny64":   64,
    "skinny128": 128,
    "craft":      64,
    "pyjamask":   96,
    "present":    64,
}

# Gohr-style fixed input difference per cipher
DELTA_P = {
    "simon32":   0x40000000,
    "gift64":    0x0000000000000001,
    "gift128":   0x00000000000000000000000000000001,
    "skinny64":  0x0000000000000001,
    "skinny128": 0x00000000000000000000000000000001,
    "craft":     0x0000000000000001,
    "pyjamask":  0x000000000000000000000001,
    "present":   0x0000000000000001,
}

FULL_ROUNDS = {
    "simon32":   32,
    "gift64":    28,
    "gift128":   40,
    "skinny64":  32,
    "skinny128": 40,
    "craft":     32,
    "pyjamask":  14,
    "present":   31,
}

N_TRAIN  = 1_000_000
N_VAL    =    50_000
N_TEST   =    50_000
BATCH    =     5_000
EPOCHS   =        10


def get_cipher(name: str):
    """Return an instantiated cipher by name."""
    from neural_cryptanalysis.ciphers.simon    import SimonCipher
    from neural_cryptanalysis.ciphers.gift64   import Gift64Cipher
    from neural_cryptanalysis.ciphers.gift128  import Gift128Cipher
    from neural_cryptanalysis.ciphers.skinny64 import Skinny64Cipher
    from neural_cryptanalysis.ciphers.skinny128 import Skinny128Cipher
    from neural_cryptanalysis.ciphers.craft    import CraftCipher
    from neural_cryptanalysis.ciphers.pyjamask import PyjamaskCipher
    from neural_cryptanalysis.ciphers.present  import PresentCipher

    registry = {
        "simon32":   SimonCipher,
        "gift64":    Gift64Cipher,
        "gift128":   Gift128Cipher,
        "skinny64":  Skinny64Cipher,
        "skinny128": Skinny128Cipher,
        "craft":     CraftCipher,
        "pyjamask":  PyjamaskCipher,
        "present":   PresentCipher,
    }

    if name not in registry:
        raise ValueError(f"Unknown cipher '{name}'. Available: {list(registry)}")

    return registry[name]()
