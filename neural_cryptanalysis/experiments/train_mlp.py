"""
Train MLP distinguisher.

Usage:
    python -m neural_cryptanalysis.experiments.train_mlp \
        --dataset neural_cryptanalysis/data/datasets/simon_delta_r4 \
        --epochs 15 --batch 256 --lr 3e-4

The dataset must have been generated and saved via generate_dataset() + save_dataset().
Works with any representation (delta, concat, bitslice, statistical, joint).
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from neural_cryptanalysis.models.mlp import MLP
from neural_cryptanalysis.data.generator import load_dataset


def parse_args():
    p = argparse.ArgumentParser(description="Train MLP distinguisher")
    p.add_argument("--dataset", type=str,
                   default="neural_cryptanalysis/data/datasets/simon_delta_r4",
                   help="Path prefix to dataset (without _X.npy / _y.npy)")
    p.add_argument("--epochs",  type=int,   default=15)
    p.add_argument("--batch",   type=int,   default=256)
    p.add_argument("--lr",      type=float, default=3e-4)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--save",    type=str,   default=None,
                   help="Path to save trained model weights (.pt)")
    return p.parse_args()


def train(args):
    # ── Load data ──────────────────────────────────────────────────────────────
    X, y = load_dataset(args.dataset)

    # Flatten if needed (e.g. bitslice returns 2D)
    if X.ndim > 2:
        X = X.reshape(len(X), -1)

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
    model = MLP(input_dim=X.shape[1], dropout=args.dropout)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MLP  input_dim={X.shape[1]}  params={total_params:,}")

    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training loop ──────────────────────────────────────────────────────────
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
        print(f"Epoch {epoch + 1:>2}/{args.epochs}  Loss: {avg_loss:.4f}")

    # ── Evaluation ─────────────────────────────────────────────────────────────
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for xb, yb in test_loader:
            preds     = model(xb)
            predicted = (preds > 0.5).float()
            correct  += (predicted == yb).sum().item()
            total    += yb.size(0)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

    # ── Save ───────────────────────────────────────────────────────────────────
    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        torch.save(model.state_dict(), args.save)
        print(f"Model saved to {args.save}")

    return model, accuracy


if __name__ == "__main__":
    args = parse_args()
    train(args)
