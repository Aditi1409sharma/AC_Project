import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from neural_cryptanalysis.models.mlp import MLP
from neural_cryptanalysis.data.generator import load_dataset

# Load dataset
X, y = load_dataset("neural_cryptanalysis/data/datasets/simon_delta_r4")

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# Train-test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=128)

# Model
model = MLP(input_dim=X.shape[1])

# Loss + optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(10):
    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        preds = model(xb)
        loss = criterion(preds, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Evaluation
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for xb, yb in test_loader:
        preds = model(xb)
        predicted = (preds > 0.5).float()
        correct += (predicted == yb).sum().item()
        total += yb.size(0)

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")