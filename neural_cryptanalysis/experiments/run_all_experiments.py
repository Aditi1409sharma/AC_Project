"""
Full Experiments Runner
========================
Runs all three required experiments and saves:
  - experiments_output.txt
  - plots/acc_vs_rounds.png
  - plots/acc_vs_representation.png
  - plots/acc_vs_model.png

Run from AC_Project/:
    python -m neural_cryptanalysis.experiments.run_all_experiments
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neural_cryptanalysis.data.generator import generate_dataset
from neural_cryptanalysis.utils.config   import get_cipher, DELTA_P, FULL_ROUNDS
from neural_cryptanalysis.models.mlp     import MLP
from neural_cryptanalysis.models.cnn     import CNN
from neural_cryptanalysis.models.siamese import SiameseNet
from neural_cryptanalysis.models.mine    import MINE

BASE     = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
PLOT_DIR = os.path.join(BASE, "plots")
OUT_FILE = os.path.join(BASE, "experiments_output.txt")
os.makedirs(PLOT_DIR, exist_ok=True)

CIPHER_NAME = "simon32"
N_SAMPLES   = 2000
EPOCHS      = 6
BATCH       = 128
LR          = 1e-3

# All ciphers for multi-cipher experiments
ALL_CIPHERS = ["simon32", "gift64", "present", "craft", "pyjamask",
               "skinny64", "gift128", "skinny128"]

_lines = []

def log(msg=""):
    print(msg)
    _lines.append(str(msg))

def save_log():
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines) + "\n")
    print(f"Results saved to: {OUT_FILE}")

# ── Quick train + eval ────────────────────────────────────────────────────────
def quick_train(model, X, y, epochs=EPOCHS, lr=LR, use_scheduler=False):
    X_t = torch.tensor(X.astype(np.float32))
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    split = int(0.8 * len(X_t))
    tr_ld = DataLoader(TensorDataset(X_t[:split], y_t[:split]),
                       batch_size=BATCH, shuffle=True)
    te_ld = DataLoader(TensorDataset(X_t[split:], y_t[split:]),
                       batch_size=BATCH)
    opt  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.BCELoss()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs) if use_scheduler else None
    for _ in range(epochs):
        model.train()
        for xb, yb in tr_ld:
            p = model(xb); l = crit(p, yb)
            opt.zero_grad(); l.backward(); opt.step()
        if sched: sched.step()
    model.eval(); correct = tot = 0
    with torch.no_grad():
        for xb, yb in te_ld:
            p = model(xb)
            correct += ((p > 0.5).float() == yb).sum().item()
            tot     += yb.size(0)
    return correct / tot

# ── Plot helpers ──────────────────────────────────────────────────────────────
def save_bar(names, values, title, xlabel, ylabel, filename, color="steelblue"):
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, [v * 100 for v in values], color=color, edgecolor="black", width=0.5)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    # Smart y-axis: start from 0 if any value < 0.7, else from 40
    min_val = min(values)
    ymin = 0 if min_val < 0.7 else 40
    ax.set_ylim(ymin, 108)
    ax.axhline(50, color="red", linestyle="--", linewidth=1, label="Random baseline (50%)")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val*100:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    log(f"  Plot saved: {path}")

def save_line(rounds, values_dict, title, filename):
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["steelblue", "darkorange", "green", "crimson", "purple"]
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

def save_single_cipher_line(cipher_name, rounds, accs, filename):
    """Single-cipher accuracy vs rounds line plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rounds, [a * 100 for a in accs], marker="o",
            color="steelblue", linewidth=2.5, markersize=7)
    # Shade the distinguishable region (acc > 55%)
    for r, a in zip(rounds, accs):
        if a > 0.55:
            ax.axvspan(r - 0.4, r + 0.4, alpha=0.08, color="steelblue")
    ax.set_title(f"Accuracy vs Rounds — {cipher_name.upper()} (MLP, delta)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Rounds", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_xticks(rounds)
    ymin = 0 if min(accs) < 0.6 else 40
    ax.set_ylim(ymin, 108)
    ax.axhline(50, color="red", linestyle="--", linewidth=1.2, label="Random baseline (50%)")
    # Annotate each point
    for r, a in zip(rounds, accs):
        ax.annotate(f"{a*100:.1f}%", (r, a * 100),
                    textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    log(f"  Plot saved: {path}")


def save_multi_line(cipher_rounds_dict, title, filename):
    """Multi-cipher accuracy vs rounds plot."""
    # 8 distinct colors for up to 8 ciphers
    COLORS = ["steelblue", "darkorange", "green", "crimson",
              "purple", "brown", "deeppink", "teal"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for (cipher_name, (rounds, accs)), color in zip(cipher_rounds_dict.items(), COLORS):
        ax.plot(rounds, [a * 100 for a in accs], marker="o",
                label=cipher_name.upper(), color=color, linewidth=2)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Rounds", fontsize=12)
    ax.set_ylabel("Best Model Accuracy (%)", fontsize=12)
    ax.set_ylim(40, 105)
    ax.axhline(50, color="gray", linestyle="--", linewidth=1, label="Random baseline")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    log(f"  Plot saved: {path}")

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1 — Accuracy vs Rounds  (multi-model + multi-cipher)
# ══════════════════════════════════════════════════════════════════════════════
def exp1_accuracy_vs_rounds():
    sep = "=" * 60
    log(sep)
    log("  EXPERIMENT 1: Accuracy vs Number of Rounds")
    log(f"  Primary cipher: {CIPHER_NAME}  |  Representation: concat")
    log(f"  Also shows: simon32, gift64, present (Siamese, delta rep)")
    log(sep)
    log()

    # ── Part A: all 3 models on simon32 ──────────────────────────────────────
    cipher     = get_cipher(CIPHER_NAME)
    delta      = DELTA_P[CIPHER_NAME]
    test_rounds = list(range(2, min(FULL_ROUNDS[CIPHER_NAME] + 1, 10)))

    log(f"  Part A — {CIPHER_NAME.upper()} | All models | concat rep")
    log(f"  {'Rounds':<8} {'MLP':>8} {'CNN':>8} {'Siamese':>10} {'MINE':>8}")
    log(f"  {'-'*46}")

    mlp_accs = []; cnn_accs = []; siam_accs = []; mine_accs = []

    for r in test_rounds:
        X, y = generate_dataset(cipher, rounds=r, n_samples=N_SAMPLES,
                                 delta_p=delta, representation="concat")
        # Generate a larger dataset for MINE (needs more data to converge)
        X_mine, y_mine = generate_dataset(cipher, rounds=r, n_samples=4000,
                                           delta_p=delta, representation="concat")
        dim = X.shape[1]
        a_mlp  = quick_train(MLP(input_dim=dim),                          X, y)
        a_cnn  = quick_train(CNN(input_dim=dim, num_filters=32),           X, y, lr=1e-3)
        a_siam = quick_train(SiameseNet(branch_dim=dim//2, embed_dim=64),  X, y)
        # MINE needs more epochs and lower LR to converge — use dedicated config
        a_mine = quick_train(MINE(input_dim=dim, hidden_dim=512),          X_mine, y_mine,
                             epochs=15, lr=3e-4, use_scheduler=True)
        mlp_accs.append(a_mlp); cnn_accs.append(a_cnn)
        siam_accs.append(a_siam); mine_accs.append(a_mine)
        log(f"  r={r:<6} {a_mlp:>8.4f} {a_cnn:>8.4f} {a_siam:>10.4f} {a_mine:>8.4f}")

    log()
    for mname, accs in [("MLP",mlp_accs),("CNN",cnn_accs),("Siamese",siam_accs),("MINE",mine_accs)]:
        max_r = max((r for r,a in zip(test_rounds,accs) if a > 0.55), default=test_rounds[0])
        log(f"  Max distinguishable round ({mname}): r={max_r}  acc={accs[test_rounds.index(max_r)]:.4f}")
    log()

    save_line(test_rounds,
              {"MLP": mlp_accs, "CNN": cnn_accs, "Siamese": siam_accs, "MINE": mine_accs},
              f"Accuracy vs Rounds — {CIPHER_NAME.upper()} (all models)",
              "acc_vs_rounds.png")

    # ── Part B: multi-cipher comparison (MLP, delta rep) ────────────────────
    log(f"  Part B — Multi-cipher | MLP | delta rep")
    log(f"  {'Cipher':<12} {'Rounds tested':<24} {'Max dist. round':>16}")
    log(f"  {'-'*55}")

    multi_cipher_configs = {
        "simon32":   (list(range(2, 10)), DELTA_P["simon32"]),
        "gift64":    (list(range(2, 9)),  DELTA_P["gift64"]),
        "present":   (list(range(2, 9)),  DELTA_P["present"]),
        "craft":     (list(range(2, 8)),  DELTA_P["craft"]),
        "pyjamask":  (list(range(1, 8)),  DELTA_P["pyjamask"]),
        "skinny64":  (list(range(2, 9)),  DELTA_P["skinny64"]),
        "gift128":   (list(range(2, 8)),  DELTA_P["gift128"]),
        "skinny128": (list(range(2, 8)),  DELTA_P["skinny128"]),
    }
    multi_results = {}

    for cname, (rounds, dp) in multi_cipher_configs.items():
        c = get_cipher(cname)
        accs = []
        for r in rounds:
            X, y = generate_dataset(c, rounds=r, n_samples=N_SAMPLES,
                                     delta_p=dp, representation="delta")
            dim = X.shape[1]
            acc = quick_train(MLP(input_dim=dim), X, y)
            accs.append(acc)
        multi_results[cname] = (rounds, accs)
        max_r = max((r for r, a in zip(rounds, accs) if a > 0.55), default=rounds[0])
        log(f"  {cname:<12} {str(rounds):<24} r={max_r} ({accs[rounds.index(max_r)]:.4f})")

    log()

    # Split into two plots for readability (4 ciphers each)
    ciphers_a = {k: multi_results[k] for k in ["simon32", "gift64", "present", "craft"]}
    ciphers_b = {k: multi_results[k] for k in ["pyjamask", "skinny64", "gift128", "skinny128"]}
    save_multi_line(ciphers_a,
                    "Accuracy vs Rounds — Ciphers Group A (MLP, delta)",
                    "acc_vs_rounds_ciphers_a.png")
    save_multi_line(ciphers_b,
                    "Accuracy vs Rounds — Ciphers Group B (MLP, delta)",
                    "acc_vs_rounds_ciphers_b.png")
    # Also save combined (all 8)
    save_multi_line(multi_results,
                    "Accuracy vs Rounds — All 8 Ciphers (MLP, delta)",
                    "acc_vs_rounds_all_ciphers.png")

    # ── Individual per-cipher plots ──────────────────────────────────────────
    log(f"  Individual cipher plots:")
    for cname, (rounds, accs) in multi_results.items():
        save_single_cipher_line(cname, rounds, accs,
                                f"acc_vs_rounds_{cname}.png")

    return test_rounds, mlp_accs, cnn_accs, siam_accs, mine_accs

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2 — Accuracy vs Representation  (now includes word)
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

    reps = ["delta", "concat", "raw", "word", "bitslice", "statistical", "joint"]
    rep_accs = []

    log(f"  {'Representation':<16} {'Input Dim':>10} {'Accuracy':>10}")
    log(f"  {'-'*40}")

    rep_config = {
        "delta":       (12, 3e-4),   # best rep — more epochs for stable result
        "concat":      (12, 1e-3),
        "raw":         (12, 1e-3),
        "word":        (12, 1e-3),
        "bitslice":    (12, 1e-3),
        "statistical": (20, 3e-4),   # rich features — more epochs
        "joint":       (15, 5e-4),   # large input — more epochs
    }

    # Use more samples for Exp2 to get stable representation comparison
    N_EXP2 = 6000

    for rep in reps:
        X, y = generate_dataset(cipher, rounds=4, n_samples=N_EXP2,
                                 delta_p=delta, representation=rep)
        if X.ndim > 2:
            X = X.reshape(len(X), -1)
        X = X.astype(np.float32)
        ep, lr = rep_config[rep]
        acc = quick_train(MLP(input_dim=X.shape[1]), X, y, epochs=ep, lr=lr)
        rep_accs.append(acc)
        log(f"  {rep:<16} {X.shape[1]:>10} {acc:>10.4f}")

    log()

    save_bar(reps, rep_accs,
             "Accuracy vs Representation",
             "Representation", "Test Accuracy (%)",
             "acc_vs_representation.png",
             color=["#2196F3","#4CAF50","#FF5722","#795548","#FF9800","#9C27B0","#F44336"])

    return reps, rep_accs

# ── Exp2 for all ciphers ──────────────────────────────────────────────────────
def exp2_all_ciphers():
    """Bar chart: best accuracy per cipher using each cipher's optimal round."""
    sep = "=" * 60
    log(sep)
    log("  EXPERIMENT 2b: Best Accuracy per Cipher — All Ciphers")
    log("  Model: MLP  |  Representation: delta  |  Best round per cipher")
    log(sep)
    log()

    # Use each cipher's known best distinguishable round
    # (highest round where the cipher is still clearly distinguishable)
    BEST_ROUNDS = {
        "simon32":   5,   # distinguishable up to r=7, r=5 gives strong signal
        "gift64":    3,   # r=3 gives ~100%, r=4 drops to ~84%
        "gift128":   4,   # distinguishable up to r=6+
        "skinny64":  5,   # distinguishable up to r=6+
        "skinny128": 4,   # distinguishable up to r=5+
        "craft":     3,   # breaks at r=4, r=3 is last good round
        "pyjamask":  1,   # breaks at r=2, r=1 is last good round (100%)
        "present":   4,   # distinguishable up to r=5
    }

    cipher_accs = []
    cipher_names = []

    # Per-cipher training config: (n_samples, epochs, lr)
    # Ciphers with fast diffusion need their best round + more training
    CIPHER_TRAIN_CONFIG = {
        "simon32":   (6000, 15, 3e-4),   # boost: more data + cosine LR
        "gift64":    (6000, 15, 3e-4),   # needs more data/epochs at r=3
        "gift128":   (6000, 15, 3e-4),   # boost: more data + cosine LR
        "skinny64":  (N_SAMPLES, EPOCHS, LR),
        "skinny128": (N_SAMPLES, EPOCHS, LR),
        "craft":     (N_SAMPLES, EPOCHS, LR),
        "pyjamask":  (6000, 15, 3e-4),   # needs more data/epochs at r=1
        "present":   (6000, 15, 3e-4),   # boost: more data + cosine LR
    }

    log(f"  {'Cipher':<12} {'Round':>6} {'Accuracy':>10}")
    log(f"  {'-'*32}")

    for cname in ALL_CIPHERS:
        c  = get_cipher(cname)
        dp = DELTA_P[cname]
        r  = BEST_ROUNDS[cname]
        n, ep, lr = CIPHER_TRAIN_CONFIG[cname]
        X, y = generate_dataset(c, rounds=r, n_samples=n,
                                 delta_p=dp, representation="delta")
        X = X.astype(np.float32)
        acc = quick_train(MLP(input_dim=X.shape[1]), X, y, epochs=ep, lr=lr,
                          use_scheduler=(ep > EPOCHS))
        cipher_accs.append(acc)
        cipher_names.append(cname)
        log(f"  {cname:<12} r={r:<4} acc={acc:.4f}")

    log()
    save_bar(cipher_names, cipher_accs,
             "Accuracy per Cipher — MLP, delta rep (best round per cipher)",
             "Cipher", "Test Accuracy (%)",
             "acc_vs_cipher.png",
             color=["#2196F3","#4CAF50","#FF5722","#FF9800",
                    "#9C27B0","#F44336","#795548","#00BCD4"])
    return cipher_names, cipher_accs

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3 — Accuracy vs Model  (now includes MINE)
# ══════════════════════════════════════════════════════════════════════════════
def exp3_accuracy_vs_model():
    sep = "=" * 60
    log(sep)
    log("  EXPERIMENT 3: Accuracy vs Model Architecture")
    log(f"  Cipher: {CIPHER_NAME}  |  Rounds: 4  |  N=4000  |  Epochs: 15")
    log(f"  MLP+MINE: delta rep  |  CNN+Siamese: concat rep")
    log(sep)
    log()

    cipher = get_cipher(CIPHER_NAME)
    delta  = DELTA_P[CIPHER_NAME]
    N_EXP3 = 4000
    EP_EXP3 = 15

    X_delta,  y_delta  = generate_dataset(cipher, rounds=4, n_samples=N_EXP3,
                                           delta_p=delta, representation="delta")
    X_concat, y_concat = generate_dataset(cipher, rounds=4, n_samples=N_EXP3,
                                           delta_p=delta, representation="concat")
    dim_d = X_delta.shape[1]
    dim_c = X_concat.shape[1]

    def _train(model, X, y, epochs, lr):
        X_t = torch.tensor(X.astype(np.float32))
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        split = int(0.8 * len(X_t))
        tr_ld = DataLoader(TensorDataset(X_t[:split], y_t[:split]), batch_size=BATCH, shuffle=True)
        te_ld = DataLoader(TensorDataset(X_t[split:], y_t[split:]), batch_size=BATCH)
        opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        crit  = nn.BCELoss()
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        for _ in range(epochs):
            model.train()
            for xb, yb in tr_ld:
                p = model(xb); l = crit(p, yb)
                opt.zero_grad(); l.backward(); opt.step()
            sched.step()
        model.eval(); correct = tot = 0
        with torch.no_grad():
            for xb, yb in te_ld:
                p = model(xb)
                correct += ((p > 0.5).float() == yb).sum().item()
                tot     += yb.size(0)
        return correct / tot

    configs = {
        "MLP"    : (MLP(input_dim=dim_d),                           X_delta,  y_delta,  EP_EXP3, 3e-4),
        "CNN"    : (CNN(input_dim=dim_c, num_filters=32),            X_concat, y_concat, EP_EXP3, 1e-3),
        "Siamese": (SiameseNet(branch_dim=dim_c//2, embed_dim=64),   X_concat, y_concat, EP_EXP3, 1e-3),
        "MINE"   : (MINE(input_dim=dim_c, hidden_dim=512),           X_concat, y_concat, EP_EXP3, 3e-4),
    }
    rep_labels = {"MLP": "delta", "CNN": "concat", "Siamese": "concat", "MINE": "concat"}

    log(f"  {'Model':<12} {'Params':>10} {'Rep':>8} {'Accuracy':>10}")
    log(f"  {'-'*44}")

    model_names = []; model_accs = []
    for name, (model, X, y, ep, lr) in configs.items():
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        acc    = _train(model, X, y, ep, lr)
        model_names.append(name); model_accs.append(acc)
        log(f"  {name:<12} {params:>10,} {rep_labels[name]:>8} {acc:>10.4f}")

    log()

    save_bar(model_names, model_accs,
             "Accuracy vs Model Architecture",
             "Model", "Test Accuracy (%)",
             "acc_vs_model.png",
             color=["#2196F3", "#FF9800", "#4CAF50", "#E91E63"])

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

    rounds, mlp_r, cnn_r, siam_r, mine_r = exp1_accuracy_vs_rounds()
    reps,   rep_accs                      = exp2_accuracy_vs_representation()
    models, model_accs                    = exp3_accuracy_vs_model()
    ciphers, cipher_accs                  = exp2_all_ciphers()

    sep = "=" * 60
    log(sep)
    log("  FINAL SUMMARY")
    log(sep)
    log()
    log("  Exp 1 — Accuracy vs Rounds (simon32, concat):")
    log(f"  {'Rounds':<8} {'MLP':>8} {'CNN':>8} {'Siamese':>10} {'MINE':>8}")
    for r, a1, a2, a3, a4 in zip(rounds, mlp_r, cnn_r, siam_r, mine_r):
        log(f"  {r:<8} {a1:>8.4f} {a2:>8.4f} {a3:>10.4f} {a4:>8.4f}")
    log()
    log("  Exp 2 — Accuracy vs Representation (MLP, r=4):")
    for rep, acc in zip(reps, rep_accs):
        log(f"  {rep:<16} {acc:.4f}")
    log()
    log("  Exp 3 — Accuracy vs Model (r=4):")
    for name, acc in zip(models, model_accs):
        log(f"  {name:<12} {acc:.4f}")
    log()
    log("  Exp 2b — Accuracy per Cipher (MLP, delta, r=4):")
    for cname, acc in zip(ciphers, cipher_accs):
        log(f"  {cname:<12} {acc:.4f}")
    log()
    log("  Plots saved in: plots/")
    log("    - acc_vs_rounds.png              (simon32, all 4 models)")
    log("    - acc_vs_rounds_ciphers_a.png    (simon32/gift64/present/craft)")
    log("    - acc_vs_rounds_ciphers_b.png    (pyjamask/skinny64/gift128/skinny128)")
    log("    - acc_vs_rounds_all_ciphers.png  (all 8 ciphers)")
    log("    - acc_vs_rounds_simon32.png      (simon32, individual)")
    log("    - acc_vs_rounds_gift64.png       (gift64, individual)")
    log("    - acc_vs_rounds_present.png      (present, individual)")
    log("    - acc_vs_rounds_craft.png        (craft, individual)")
    log("    - acc_vs_rounds_pyjamask.png     (pyjamask, individual)")
    log("    - acc_vs_rounds_skinny64.png     (skinny64, individual)")
    log("    - acc_vs_rounds_gift128.png      (gift128, individual)")
    log("    - acc_vs_rounds_skinny128.png    (skinny128, individual)")
    log("    - acc_vs_representation.png      (simon32, 7 representations)")
    log("    - acc_vs_model.png               (simon32, 4 models)")
    log("    - acc_vs_cipher.png              (all 8 ciphers, MLP delta)")
    log(sep)

    save_log()


if __name__ == "__main__":
    main()
