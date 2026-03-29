"""
Benchmark all three models (MLP, CNN, Siamese) and save results to a txt file.
Run from AC_Project-main/:
    python -m neural_cryptanalysis.experiments.benchmark_models
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from neural_cryptanalysis.data.generator import generate_dataset
from neural_cryptanalysis.utils.config   import get_cipher, DELTA_P
from neural_cryptanalysis.models.mlp     import MLP
from neural_cryptanalysis.models.cnn     import CNN
from neural_cryptanalysis.models.siamese import SiameseNet

# ── Config ────────────────────────────────────────────────────────────────────
CIPHER_NAME    = "simon32"
ROUNDS         = 4
N_SAMPLES      = 8000       # more data -> better generalisation
EPOCHS         = 15
BATCH          = 256
LR             = 3e-4
WEIGHT_DECAY   = 1e-4
LABEL_SMOOTH   = 0.05       # prevents overconfident predictions
OUTPUT_FILE    = "model_benchmark_output.txt"

# ── Logging ───────────────────────────────────────────────────────────────────
_lines = []

def log(msg=""):
    print(msg)
    _lines.append(str(msg))

def save_log(path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines) + "\n")
    print(f"\nReport saved to: {path}")


# ── Label-smoothed BCE ────────────────────────────────────────────────────────
class SmoothBCELoss(nn.Module):
    def __init__(self, smoothing: float = 0.05):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        target_smooth = target * (1 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(pred, target_smooth)


# ── LR warmup + cosine scheduler ─────────────────────────────────────────────
def make_scheduler(opt, epochs, warmup=2):
    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        progress = (ep - warmup) / max(1, epochs - warmup)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


# ── Training ──────────────────────────────────────────────────────────────────
def run(name, model, tr_loader, te_loader, epochs=EPOCHS, smooth=True, lr=LR):
    crit   = SmoothBCELoss(LABEL_SMOOTH) if smooth else nn.BCELoss()
    opt    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    sched  = make_scheduler(opt, epochs, warmup=2)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    log(f"  Trainable parameters : {params:,}")
    log(f"  {'Epoch':<8} {'Loss':>10} {'Accuracy':>12}")
    log(f"  {'-'*34}")

    best_acc = 0.0

    for ep in range(epochs):
        model.train()
        loss_sum = 0.0
        for xb, yb in tr_loader:
            pred = model(xb)
            loss = crit(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item()
        sched.step()

        model.eval()
        correct = tot = 0
        with torch.no_grad():
            for xb, yb in te_loader:
                pred     = model(xb)
                correct += ((pred > 0.5).float() == yb).sum().item()
                tot     += yb.size(0)

        avg_loss = loss_sum / len(tr_loader)
        acc      = correct / tot
        best_acc = max(best_acc, acc)
        log(f"  {ep+1:<8} {avg_loss:>10.4f} {acc:>11.4f}")

    log(f"  {'':8} {'':>10} {'':>12}")
    log(f"  Best Test Accuracy   : {best_acc:.4f}")
    return best_acc


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    sep = "=" * 60

    log(sep)
    log("  MODEL BENCHMARK REPORT")
    log("  Neural Cryptanalysis Project")
    log(sep)
    log()
    log(f"  Cipher         : {CIPHER_NAME}")
    log(f"  Rounds         : {ROUNDS}")
    log(f"  Samples        : {N_SAMPLES}")
    log(f"  Batch size     : {BATCH}")
    log(f"  MLP/Siamese    : LR={LR}, Epochs={EPOCHS}, label_smooth=0.05/none")
    log(f"  CNN            : LR=1e-3, Epochs=10, label_smooth=none")
    log(f"  Representation : concat  (bits(C) || bits(C'))")
    log()

    log("Generating dataset...")
    cipher = get_cipher(CIPHER_NAME)
    delta  = DELTA_P[CIPHER_NAME]
    X, y   = generate_dataset(cipher, rounds=ROUNDS, n_samples=N_SAMPLES,
                               delta_p=delta, representation="concat")

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    split    = int(0.8 * len(X_t))
    X_tr, X_te = X_t[:split], X_t[split:]
    y_tr, y_te = y_t[:split], y_t[split:]

    tr_ld = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH, shuffle=True)
    te_ld = DataLoader(TensorDataset(X_te, y_te), batch_size=BATCH)

    input_dim  = X_t.shape[1]
    branch_dim = input_dim // 2

    log(f"  Dataset shape  : X={tuple(X_t.shape)}  y={tuple(y_t.shape)}")
    log(f"  Label balance  : {y.mean():.2f}")
    log(f"  input_dim={input_dim}  branch_dim={branch_dim}")
    log()

    results = {}

    # MLP
    log(sep)
    log("  MODEL 1: MLP")
    log(sep)
    log("  Architecture : Linear(1024)->LN->GELU x5 + 2 skip connections -> Linear(1)")
    log()
    results["MLP"] = run("MLP", MLP(input_dim=input_dim), tr_ld, te_ld)
    log()

    # CNN — no smoothing, higher LR, more epochs, larger model
    log(sep)
    log("  MODEL 2: CNN")
    log(sep)
    log("  Architecture : 3-ch (C,C',XOR) stem -> MultiScale -> ResStages -> avg+max+std -> head")
    log("  Loss         : BCELoss (no smoothing)")
    log("  LR           : 1e-3  Epochs: 10")
    log()
    results["CNN"] = run("CNN", CNN(input_dim=input_dim, num_filters=32),
                         tr_ld, te_ld, epochs=10, smooth=False, lr=1e-3)
    log()

    # Siamese — plain BCE, original architecture, embed_dim=64
    log(sep)
    log("  MODEL 3: Siamese Network")
    log(sep)
    log("  Architecture : Shared encoder(2L) -> concat + |diff| -> classifier")
    log("  Loss         : BCELoss (no smoothing — allows 100% accuracy)")
    log()
    results["Siamese"] = run("Siamese", SiameseNet(branch_dim=branch_dim, embed_dim=64),
                             tr_ld, te_ld, smooth=False)
    log()

    # Summary
    log(sep)
    log("  SUMMARY")
    log(sep)
    log(f"  {'Model':<12} {'Best Accuracy':>14}  {'vs Baseline':>12}")
    log(f"  {'-'*42}")
    for name, acc in results.items():
        gain = acc - 0.5
        log(f"  {name:<12} {acc:>14.4f}  {gain:>+11.4f}")
    log()
    log(f"  Baseline (random guessing) : 0.5000")
    log(f"  All models above baseline  : {all(v > 0.5 for v in results.values())}")
    log(sep)

    out_path = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", OUTPUT_FILE)
    )
    save_log(out_path)


if __name__ == "__main__":
    main()
