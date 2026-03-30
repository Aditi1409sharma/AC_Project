"""
Train MINE distinguisher on a ciphertext-pair dataset.

Usage:
    python -m neural_cryptanalysis.experiments.train_mine \
        --dataset neural_cryptanalysis/data/datasets/simon_concat_r4 \
        --epochs 15 --batch 256 --lr 3e-4

The dataset must have been generated with representation='concat'
so that input shape is (2 * block_bits,).

MINE internally augments the input with the XOR difference channel:
    [C || C' || ΔC]  →  shape (3 * block_bits,)
giving the statistics network the differential signal directly.
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from neural_cryptanalysis.models.mine    import MINE
from neural_cryptanalysis.data.generator import load_dataset


def parse_args():
    p = argparse.ArgumentParser(description="Train MINE distinguisher")
    p.add_argument("--dataset",    type=str,
                   default="neural_cryptanalysis/data/datasets/simon_concat_r4",
                   help="Path prefix to dataset (without _X.npy / _y.npy)")
    p.add_argument("--epochs",     type=int,   default=15)
    p.add_argument("--batch",      type=int,   default=256)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--hidden_dim", type=int,   default=512,
                   help="Hidden dimension of the statistics network")
    p.add_argument("--save",       type=str,   default=None,
                   help="Path to save trained model weights (.pt)")
    return p.parse_args()


def train(args):
    # ── Load data ──────────────────────────────────────────────────────────────
    X, y = load_dataset(args.dataset)

    assert X.ndim == 2 and X.shape[1] % 2 == 0, \
        f"Input dim {X.shape[1]} must be even (concat representation required)"

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=args.batch, shuffle=True)
    test_loader  = DataLoader(TensorDataset(X_test,  y_test),
                              batch_size=args.batch)

    # ── Model ──────────────────────────────────────────────────────────────────
    model = MINE(input_dim=X.shape[1], hidden_dim=args.hidden_dim)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MINE  input_dim={X.shape[1]}  hidden_dim={args.hidden_dim}  params={total_params:,}")
    print(f"      augmented input: [C || C' || ΔC]  →  dim={X.shape[1] + X.shape[1]//2}")

    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training loop ──────────────────────────────────────────────────────────
    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            preds = model(xb)
            loss  = criterion(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        # ── Per-epoch eval ─────────────────────────────────────────────────────
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                preds     = model(xb)
                predicted = (preds > 0.5).float()
                correct  += (predicted == yb).sum().item()
                total    += yb.size(0)
        acc = correct / total
        best_acc = max(best_acc, acc)
        print(f"Epoch {epoch + 1:>2}/{args.epochs}  Loss: {avg_loss:.4f}  Acc: {acc:.4f}")

    print(f"\nBest Test Accuracy: {best_acc:.4f}")

    # ── Save ───────────────────────────────────────────────────────────────────
    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        torch.save(model.state_dict(), args.save)
        print(f"Model saved to {args.save}")

    return model, best_acc


if __name__ == "__main__":
    args = parse_args()
    train(args)
