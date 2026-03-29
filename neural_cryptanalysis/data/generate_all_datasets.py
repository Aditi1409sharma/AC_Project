"""
generate_all_datasets.py
========================
Master script to generate ALL datasets for the neural cryptanalysis project.

Generates datasets for:
  - 8 ciphers  x  6 representations  x  N round counts
  - Each dataset: 100,000 samples (50k cipher + 50k random)

Run from the project root:
    python generate_all_datasets.py

Estimated time: 30-60 min depending on machine speed.
Use --cipher and --repr flags to generate subsets (see bottom of file).
"""

import os
import sys
import time
import argparse
import traceback
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"ROOT = {ROOT}")          # ← add this
print(f"Contents: {os.listdir(ROOT)}")  # ← and this
sys.path.insert(0, ROOT)

from neural_cryptanalysis.ciphers.simon     import SimonCipher
from neural_cryptanalysis.ciphers.gift64    import Gift64Cipher
from neural_cryptanalysis.ciphers.gift128   import Gift128Cipher
from neural_cryptanalysis.ciphers.present   import PresentCipher
from neural_cryptanalysis.ciphers.pyjamask  import PyjamaskCipher
from neural_cryptanalysis.ciphers.craft     import CraftCipher
from neural_cryptanalysis.ciphers.skinny64  import Skinny64Cipher
from neural_cryptanalysis.ciphers.skinny128 import Skinny128Cipher
from neural_cryptanalysis.data.generator    import generate_dataset, save_dataset

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

N_SAMPLES = 100_000          # per dataset (50k cipher + 50k random)
REPRESENTATIONS = [          # must match keys in your REPRESENTATIONS dict
    "delta",
    "raw",
    "concat",
    "bitslice",
    "joint",
    "statistical",
]

# Each cipher: (instance, name, delta_p, round_counts_to_test)
# Round counts chosen to span: very easy → borderline → hard for ML
CIPHER_CONFIGS = [
    (
        SimonCipher(),
        "simon",
        0x40000000,          # Gohr's canonical difference for Simon32/64
        [4, 5, 6, 7, 8],     # full=32; ML usually breaks ≤7
    ),
    (
        Gift64Cipher(),
        "gift64",
        0x0000000000000001,
        [4, 5, 6, 7, 8],     # full=28
    ),
    (
        Gift128Cipher(),
        "gift128",
        0x00000000000000000000000000000001,
        [4, 5, 6, 7, 8],     # full=40
    ),
    (
        PresentCipher(),
        "present",
        0x0000000000000001,
        [4, 5, 6, 7, 8],     # full=31
    ),
    (
        PyjamaskCipher(),
        "pyjamask",
        0x000000000000000000000001,
        [4, 5, 6, 7, 8],     # full=14
    ),
    (
        CraftCipher(),
        "craft",
        0x0000000000000001,
        [4, 5, 6, 7, 8],     # full=32
    ),
    (
        Skinny64Cipher(),
        "skinny64",
        0x0000000000000001,
        [4, 5, 6, 7, 8],     # full=32
    ),
    (
        Skinny128Cipher(),
        "skinny128",
        0x00000000000000000000000000000001,
        [4, 5, 6, 7, 8],     # full=40
    ),
]

# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def dataset_name(cipher_name: str, repr_name: str, rounds: int) -> str:
    """Consistent filename: e.g. simon_delta_r5"""
    return f"{cipher_name}_{repr_name}_r{rounds}"


def already_exists(name: str) -> bool:
    """Skip if both _X.npy and _y.npy already exist."""
    base = os.path.join(ROOT, "neural_cryptanalysis", "data", "datasets", name)
    return os.path.exists(base + "_X.npy") and os.path.exists(base + "_y.npy")


def fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds/60:.1f}min"


def print_header(text: str):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def print_progress(done: int, total: int, elapsed: float):
    pct  = done / total * 100
    eta  = (elapsed / done) * (total - done) if done > 0 else 0
    bar  = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
    print(f"  [{bar}] {done}/{total} ({pct:.0f}%)  elapsed={fmt_time(elapsed)}  ETA={fmt_time(eta)}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN GENERATION LOOP
# ═════════════════════════════════════════════════════════════════════════════

def generate_all(
    cipher_filter: list[str] | None = None,
    repr_filter:   list[str] | None = None,
    skip_existing: bool = True,
):
    # Count total jobs
    jobs = []
    for cipher, cname, delta_p, round_list in CIPHER_CONFIGS:
        if cipher_filter and cname not in cipher_filter:
            continue
        for repr_name in REPRESENTATIONS:
            if repr_filter and repr_name not in repr_filter:
                continue
            for rounds in round_list:
                jobs.append((cipher, cname, delta_p, repr_name, rounds))

    total    = len(jobs)
    skipped  = 0
    failed   = 0
    done     = 0
    t_start  = time.time()

    print_header(f"Generating {total} datasets  |  {N_SAMPLES:,} samples each")
    print(f"  Ciphers : {[c for _,c,_,_ in CIPHER_CONFIGS if not cipher_filter or c in cipher_filter]}")
    print(f"  Reprs   : {repr_filter or REPRESENTATIONS}")
    print(f"  Skip existing: {skip_existing}")

    results = []   # (name, status, time_s)

    for i, (cipher, cname, delta_p, repr_name, rounds) in enumerate(jobs, 1):
        name = dataset_name(cname, repr_name, rounds)

        # Skip if already generated
        if skip_existing and already_exists(name):
            print(f"  [SKIP] {name}")
            skipped += 1
            results.append((name, "skipped", 0))
            continue

        print(f"\n  [{i}/{total}] Generating: {name}")
        t0 = time.time()

        try:
            X, y = generate_dataset(
                cipher         = cipher,
                rounds         = rounds,
                n_samples      = N_SAMPLES,
                delta_p        = delta_p,
                representation = repr_name,
            )

            save_dataset(X, y, name)

            elapsed = time.time() - t0
            done   += 1
            shape_str = f"X={X.shape}  y={y.shape}"
            balance   = f"label balance: {y.mean():.2f}"
            print(f"     ✅  {shape_str}  |  {balance}  |  {fmt_time(elapsed)}")
            results.append((name, "ok", elapsed))

        except Exception as e:
            elapsed = time.time() - t0
            failed += 1
            print(f"     ❌  FAILED: {e}")
            traceback.print_exc()
            results.append((name, f"FAILED: {e}", elapsed))

        # Rolling progress bar
        total_elapsed = time.time() - t_start
        print_progress(i - skipped, total - skipped, total_elapsed)

    # ── Summary ──────────────────────────────────────────────────────────────
    total_elapsed = time.time() - t_start
    print_header("SUMMARY")
    print(f"  Total jobs   : {total}")
    print(f"  Generated    : {done}  ✅")
    print(f"  Skipped      : {skipped}  (already existed)")
    print(f"  Failed       : {failed}  {'❌' if failed else '✅'}")
    print(f"  Total time   : {fmt_time(total_elapsed)}")

    if failed:
        print("\n  Failed datasets:")
        for name, status, _ in results:
            if status.startswith("FAILED"):
                print(f"    ✗ {name}  →  {status}")

    print()
    return results


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate all neural cryptanalysis datasets."
    )
    parser.add_argument(
        "--cipher", nargs="+",
        choices=[c for _,c,_,_ in CIPHER_CONFIGS],
        help="Only generate for these ciphers (default: all)",
        default=None,
    )
    parser.add_argument(
        "--repr", nargs="+",
        choices=REPRESENTATIONS,
        help="Only generate for these representations (default: all)",
        default=None,
    )
    parser.add_argument(
        "--no-skip", action="store_true",
        help="Regenerate even if dataset already exists",
    )
    parser.add_argument(
        "--n-samples", type=int, default=N_SAMPLES,
        help=f"Samples per dataset (default: {N_SAMPLES})",
    )
    args = parser.parse_args()

    N_SAMPLES = args.n_samples   # allow override from CLI

    generate_all(
        cipher_filter = args.cipher,
        repr_filter   = args.repr,
        skip_existing = not args.no_skip,
    )