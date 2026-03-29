import numpy as np
from tqdm import tqdm

from neural_cryptanalysis.ciphers.base import BlockCipher
from neural_cryptanalysis.ciphers.random_perm import RandomPermutation
from neural_cryptanalysis.representations import REPRESENTATIONS 


def generate_dataset(
    cipher: BlockCipher,
    rounds: int,
    n_samples: int,
    delta_p: int,
    key: int | None = None,
    representation: str = "delta",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate dataset for neural cryptanalysis.

    Returns:
        X: feature array
        y: labels (1 = cipher, 0 = random permutation)
    """

    assert representation in REPRESENTATIONS, f"Unknown representation: {representation}"

    half = n_samples // 2
    bits = cipher.block_bits

    rng_perm = RandomPermutation(bits)

    if key is None:
        key = cipher.random_key()

    transform = REPRESENTATIONS[representation]

    pairs = []
    labels = []

    # =========================
    # Cipher-generated samples
    # =========================
    import os  # ← add at top

# inside both the cipher loop AND the random perm loop:
    for _ in tqdm(range(half), desc=f"Cipher r={rounds}", leave=False):
        p  = int.from_bytes(os.urandom((bits + 7) // 8), byteorder='little') & ((1 << bits) - 1)
        p_ = int(p ^ delta_p)

        c  = cipher.encrypt(p,  key, rounds)
        c_ = cipher.encrypt(p_, key, rounds)
        # ... rest unchanged

        if representation == "joint":
            x = transform(p, p_, c, c_, bits)
        else:
            x = transform(c, c_, bits)

        pairs.append(x)
        labels.append(1)

    # =========================
    # Random permutation samples
    # =========================
    rnd_key = rng_perm.random_key()

    for _ in tqdm(range(half), desc=f"Random  r={rounds}", leave=False):
        p  = int.from_bytes(os.urandom((bits + 7) // 8), byteorder='little') & ((1 << bits) - 1)
        p_ = int(p ^ delta_p)

        c = rng_perm.encrypt(p, rnd_key, rounds)
        c_ = rng_perm.encrypt(p_, rnd_key, rounds)

        if representation == "joint":
            x = transform(p, p_, c, c_, bits)
        else:
            x = transform(c, c_, bits)

        pairs.append(x)
        labels.append(0)

    # Convert to numpy arrays
    X = np.array(pairs, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)

    # Shuffle dataset
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]


# =========================
# Save / Load utilities
# =========================

import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def save_dataset(X, y, filename: str):
    """
    Save dataset inside: neural_cryptanalysis/data/datasets/
    """
    dataset_dir = os.path.join(BASE_DIR, "datasets")
    os.makedirs(dataset_dir, exist_ok=True)

    full_path = os.path.join(dataset_dir, filename)

    np.save(full_path + "_X.npy", X)
    np.save(full_path + "_y.npy", y)


def load_dataset(path: str):
    X = np.load(path + "_X.npy")
    y = np.load(path + "_y.npy")
    return X, y