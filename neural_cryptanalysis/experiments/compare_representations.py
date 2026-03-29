"""
Representation Comparison Experiment
=====================================
Trains an MLP on each representation and records accuracy.
Justifies the choice of representation for the project.

Run from AC_Project-main/:
    python -m neural_cryptanalysis.experiments.compare_representations

Output saved to: representation_comparison_output.txt
"""

import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from neural_cryptanalysis.data.generator import generate_dataset
from neural_cryptanalysis.utils.config   import get_cipher, DELTA_P
from neural_cryptanalysis.models.mlp     import MLP

# ── Config ────────────────────────────────────────────────────────────────────
CIPHER_NAME = "simon32"
ROUNDS      = 4
N_SAMPLES   = 6000
EPOCHS      = 10
BATCH       = 128
LR          = 1e-3
OUTPUT_FILE = "representation_comparison_output.txt"

# ── Logging ───────────────────────────────────────────────────────────────────
_lines = []

def log(msg=""):
    print(msg)
    _lines.append(str(msg))

def save_log(path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines) + "\n")
    print(f"\nSaved to: {path}")

# ── Training helper ───────────────────────────────────────────────────────────
def train_and_eval(X, y, epochs=EPOCHS, lr=LR):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    split    = int(0.8 * len(X_t))
    X_tr, X_te = X_t[:split], X_t[split:]
    y_tr, y_te = y_t[:split], y_t[split:]

    tr_ld = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH, shuffle=True)
    te_ld = DataLoader(TensorDataset(X_te, y_te), batch_size=BATCH)

    model = MLP(input_dim=X_t.shape[1])
    crit  = nn.BCELoss()
    opt   = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        model.train()
        for xb, yb in tr_ld:
            pred = model(xb); loss = crit(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    correct = tot = 0
    with torch.no_grad():
        for xb, yb in te_ld:
            pred     = model(xb)
            correct += ((pred > 0.5).float() == yb).sum().item()
            tot     += yb.size(0)
    return correct / tot, X_t.shape[1]

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    sep = "=" * 65

    log(sep)
    log("  REPRESENTATION COMPARISON REPORT")
    log("  Neural Cryptanalysis Project")
    log(sep)
    log()
    log(f"  Cipher  : {CIPHER_NAME}  |  Rounds : {ROUNDS}")
    log(f"  Samples : {N_SAMPLES}    |  Epochs : {EPOCHS}  |  Model: MLP")
    log()

    cipher = get_cipher(CIPHER_NAME)
    delta  = DELTA_P[CIPHER_NAME]

    representations = {
        "delta"      : "concat",   # delta uses concat key but delta rep
        "concat"     : "concat",
        "bitslice"   : "bitslice",
        "joint"      : "joint",
        "statistical": "statistical",
    }

    results = {}

    # Per-representation config: (epochs, lr)
    rep_config = {
        "delta"      : (10, 1e-3),
        "concat"     : (10, 1e-3),
        "bitslice"   : (10, 1e-3),
        "statistical": (15, 5e-4),   # more epochs, lower LR for richer features
        "joint"      : (15, 5e-4),   # more epochs for larger input
    }

    for rep_name in ["delta", "concat", "bitslice", "statistical", "joint"]:
        log(f"  Generating dataset: representation='{rep_name}'...")
        X, y = generate_dataset(cipher, rounds=ROUNDS, n_samples=N_SAMPLES,
                                 delta_p=delta, representation=rep_name)

        # joint returns shape (N, 4*bits) — already flat; others may be 2D
        if X.ndim > 2:
            X = X.reshape(len(X), -1)
        X = X.astype(np.float32)

        ep, lr = rep_config[rep_name]
        acc, dim = train_and_eval(X, y, epochs=ep, lr=lr)
        results[rep_name] = (acc, dim)
        log(f"    input_dim={dim:<6}  accuracy={acc:.4f}")
        log()

    # ── Summary table ─────────────────────────────────────────────────────────
    log(sep)
    log("  SUMMARY — Representation vs Accuracy")
    log(sep)
    log()
    log(f"  {'Representation':<16} {'Input Dim':>10} {'Accuracy':>10}  Effect / Notes")
    log(f"  {'-'*65}")

    notes = {
        "delta"      : "Best signal — directly encodes differential pattern (C XOR C')",
        "concat"     : "Full information — model learns relations freely",
        "bitslice"   : "Bit-position aligned — ideal for CNN local pattern detection",
        "statistical": "Interpretable but weak — only 3 features (HW-based)",
        "joint"      : "Richest input — includes plaintext, useful for white-box analysis",
    }

    for rep, (acc, dim) in results.items():
        log(f"  {rep:<16} {dim:>10} {acc:>10.4f}  {notes[rep]}")

    log()
    log("  Key Takeaways:")
    log("  - delta      : most aligned with classical differential cryptanalysis")
    log("  - concat     : flexible, works well with MLP and Siamese")
    log("  - bitslice   : preserves bit-position structure, best for CNN")
    log("  - statistical: interpretable but too compressed (3 features only)")
    log("  - joint      : strongest in white-box setting (plaintext visible)")
    log()
    log(f"  Baseline (random) : 0.5000")
    log(sep)

    out_path = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", OUTPUT_FILE)
    )
    save_log(out_path)


if __name__ == "__main__":
    main()
