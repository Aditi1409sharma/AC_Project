"""
Full Experiments Runner
========================
Runs all three required experiments and saves:
  - experiments_output.txt   : numerical results table
  - plots/acc_vs_rounds.png
  - plots/acc_vs_representation.png
  - plots/acc_vs_model.png

Run from AC_Project-main/:
    python -m neural_cryptanalysis.experiments.run_all_experiments
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (no display needed)
import matplotlib.pyplot as plt

from neural_cryptanalysis.data.generator import generate_dataset
from neural_cryptanalysis.utils.config   import get_cipher, DELTA_P, FULL_ROUNDS
from neural_cryptanalysis.models.mlp     import MLP
from neural_cryptanalysis.models.cnn     import CNN
from neural_cryptanalysis.models.siamese import SiameseNet

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE     = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
PLOT_DIR = os.path.join(BASE, "plots")
OUT_FILE = os.path.join(BASE, "experiments_output.txt")
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
CIPHER_NAME = "simon32"
N_SAMPLES   = 2000
EPOCHS      = 6
BATCH       = 128
LR          = 1e-3

# ── Logging ───────────────────────────────────────────────────────────────────
_lines = []

def log(msg=""):
    print(msg)
    _lines.append(str(msg))

def save_log():
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines) + "\n")
    log(f"Results saved to: {OUT_FILE}")

# ── Quick train + eval ────────────────────────────────────────────────────────
def quick_train(model, X, y, epochs=EPOCHS, lr=LR):
    X_t = torch.tensor(X.astype(np.float32))
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    split = int(0.8 * len(X_t))
    tr_ld = DataLoader(TensorDataset(X_t[:split], y_t[:split]),
                       batch_size=BATCH, shuffle=True)
    te_ld = DataLoader(TensorDataset(X_t[split:], y_t[split:]),
                       batch_size=BATCH)

    opt  = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCELoss()

    for _ in range(epochs):
        model.train()
        for xb, yb in tr_ld:
            p = model(xb); l = crit(p, yb)
            opt.zero_grad(); l.backward(); opt.step()

    model.eval(); correct = tot = 0
    with torch.no_grad():
        for xb, yb in te_ld:
            p = model(xb)
            correct += ((p > 0.5).float() == yb).sum().item()
            tot     += yb.size(0)
    return correct / tot

# ── Plot helpers ──────────────────────────────────────────────────────────────
def save_bar(names, values, title, xlabel, ylabel, filename, color="steelblue"):
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, [v * 100 for v in values], color=color, edgecolor="black", width=0.5)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_ylim(40, 105)
    ax.axhline(50, color="red", linestyle="--", linewidth=1, label="Random baseline (50%)")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val*100:.1f}%", ha="center", va="bottom", fontsize=10)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    log(f"  Plot saved: {path}")

def save_line(rounds, values_dict, title, filename):
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["steelblue", "darkorange", "green", "red"]
    for (label, vals), color in zip(values_dict.items(), colors):
        ax.plot(rounds, [v * 100 for v in vals], marker="o",
                label=label, color=color, linewidth=2)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Rounds", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_ylim(40, 105)
    ax.axhline(50, color="gray", linestyle="--", linewidth=1, label="Random baseline")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    log(f"  Plot saved: {path}")

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1 — Accuracy vs Rounds
# ══════════════════════════════════════════════════════════════════════════════
def exp1_accuracy_vs_rounds():
    sep = "=" * 60
    log(sep)
    log("  EXPERIMENT 1: Accuracy vs Number of Rounds")
    log(f"  Cipher: {CIPHER_NAME}  |  Representation: concat")
    log(sep)
    log()

    cipher     = get_cipher(CIPHER_NAME)
    delta      = DELTA_P[CIPHER_NAME]
    max_rounds = FULL_ROUNDS[CIPHER_NAME]

    # Test rounds: from 2 up to where accuracy drops near 50%
    test_rounds = list(range(2, min(max_rounds + 1, 10)))

    log(f"  {'Rounds':<8} {'MLP':>8} {'CNN':>8} {'Siamese':>10}")
    log(f"  {'-'*38}")

    mlp_accs  = []
    cnn_accs  = []
    siam_accs = []

    for r in test_rounds:
        X, y = generate_dataset(cipher, rounds=r, n_samples=N_SAMPLES,
                                 delta_p=delta, representation="concat")
        dim = X.shape[1]

        a_mlp  = quick_train(MLP(input_dim=dim), X, y)
        a_cnn  = quick_train(CNN(input_dim=dim, num_filters=32), X, y, lr=1e-3)
        a_siam = quick_train(SiameseNet(branch_dim=dim//2, embed_dim=64), X, y)

        mlp_accs.append(a_mlp)
        cnn_accs.append(a_cnn)
        siam_accs.append(a_siam)

        log(f"  r={r:<6} {a_mlp:>8.4f} {a_cnn:>8.4f} {a_siam:>10.4f}")

    log()

    # Find max distinguishable round (accuracy > 55%)
    for model_name, accs in [("MLP", mlp_accs), ("CNN", cnn_accs), ("Siamese", siam_accs)]:
        max_r = max((r for r, a in zip(test_rounds, accs) if a > 0.55), default=test_rounds[0])
        log(f"  Max distinguishable round ({model_name}): r={max_r}  acc={accs[test_rounds.index(max_r)]:.4f}")
    log()

    save_line(test_rounds,
              {"MLP": mlp_accs, "CNN": cnn_accs, "Siamese": siam_accs},
              f"Accuracy vs Rounds — {CIPHER_NAME.upper()}",
              "acc_vs_rounds.png")

    return test_rounds, mlp_accs, cnn_accs, siam_accs

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2 — Accuracy vs Representation
# ══════════════════════════════════════════════════════════════════════════════
def exp2_accuracy_vs_representation():
    sep = "=" * 60
    log(sep)
    log("  EXPERIMENT 2: Accuracy vs Representation")
    log(f"  Cipher: {CIPHER_NAME}  |  Rounds: 4  |  Model: MLP")
    log(sep)
    log()

    cipher = get_cipher(CIPHER_NAME)
    delta  = DELTA_P[CIPHER_NAME]

    reps = ["delta", "concat", "bitslice", "statistical", "joint"]
    rep_accs = []

    log(f"  {'Representation':<16} {'Input Dim':>10} {'Accuracy':>10}")
    log(f"  {'-'*40}")

    for rep in reps:
        X, y = generate_dataset(cipher, rounds=4, n_samples=N_SAMPLES,
                                 delta_p=delta, representation=rep)
        if X.ndim > 2:
            X = X.reshape(len(X), -1)
        X = X.astype(np.float32)

        ep = 15 if rep in ("statistical", "joint") else EPOCHS
        lr = 5e-4 if rep in ("statistical", "joint") else LR
        acc = quick_train(MLP(input_dim=X.shape[1]), X, y, epochs=ep, lr=lr)
        rep_accs.append(acc)
        log(f"  {rep:<16} {X.shape[1]:>10} {acc:>10.4f}")

    log()

    save_bar(reps, rep_accs,
             "Accuracy vs Representation",
             "Representation", "Test Accuracy (%)",
             "acc_vs_representation.png",
             color=["#2196F3","#4CAF50","#FF9800","#9C27B0","#F44336"])

    return reps, rep_accs

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3 — Accuracy vs Model
# ══════════════════════════════════════════════════════════════════════════════
def exp3_accuracy_vs_model():
    sep = "=" * 60
    log(sep)
    log("  EXPERIMENT 3: Accuracy vs Model Architecture")
    log(f"  Cipher: {CIPHER_NAME}  |  Rounds: 4  |  Representation: concat")
    log(sep)
    log()

    cipher = get_cipher(CIPHER_NAME)
    delta  = DELTA_P[CIPHER_NAME]

    X, y = generate_dataset(cipher, rounds=4, n_samples=N_SAMPLES,
                             delta_p=delta, representation="concat")
    dim = X.shape[1]

    models = {
        "MLP"    : MLP(input_dim=dim),
        "CNN"    : CNN(input_dim=dim, num_filters=32),
        "Siamese": SiameseNet(branch_dim=dim//2, embed_dim=64),
    }
    lrs = {"MLP": LR, "CNN": 1e-3, "Siamese": LR}

    log(f"  {'Model':<12} {'Params':>10} {'Accuracy':>10}")
    log(f"  {'-'*36}")

    model_names = []
    model_accs  = []

    for name, model in models.items():
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        acc    = quick_train(model, X, y, lr=lrs[name])
        model_names.append(name)
        model_accs.append(acc)
        log(f"  {name:<12} {params:>10,} {acc:>10.4f}")

    log()

    save_bar(model_names, model_accs,
             "Accuracy vs Model Architecture",
             "Model", "Test Accuracy (%)",
             "acc_vs_model.png",
             color=["#2196F3", "#FF9800", "#4CAF50"])

    return model_names, model_accs

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    log("=" * 60)
    log("  NEURAL CRYPTANALYSIS — FULL EXPERIMENTS")
    log(f"  Cipher: {CIPHER_NAME}  |  Samples/run: {N_SAMPLES}  |  Epochs: {EPOCHS}")
    log("=" * 60)
    log()

    rounds, mlp_r, cnn_r, siam_r = exp1_accuracy_vs_rounds()
    reps,   rep_accs              = exp2_accuracy_vs_representation()
    models, model_accs            = exp3_accuracy_vs_model()

    # ── Final summary ─────────────────────────────────────────────────────────
    sep = "=" * 60
    log(sep)
    log("  FINAL SUMMARY")
    log(sep)
    log()
    log("  Exp 1 — Accuracy vs Rounds (MLP, concat, simon32):")
    log(f"  {'Rounds':<8} {'MLP':>8} {'CNN':>8} {'Siamese':>10}")
    for r, a1, a2, a3 in zip(rounds, mlp_r, cnn_r, siam_r):
        log(f"  {r:<8} {a1:>8.4f} {a2:>8.4f} {a3:>10.4f}")
    log()
    log("  Exp 2 — Accuracy vs Representation (MLP, r=4):")
    for rep, acc in zip(reps, rep_accs):
        log(f"  {rep:<16} {acc:.4f}")
    log()
    log("  Exp 3 — Accuracy vs Model (concat, r=4):")
    for name, acc in zip(models, model_accs):
        log(f"  {name:<12} {acc:.4f}")
    log()
    log("  Plots saved in: plots/")
    log("    - acc_vs_rounds.png")
    log("    - acc_vs_representation.png")
    log("    - acc_vs_model.png")
    log(sep)

    save_log()


if __name__ == "__main__":
    main()
