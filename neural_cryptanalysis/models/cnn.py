import torch
import torch.nn as nn


class _ResBlock(nn.Module):
    """1-D residual conv block."""
    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class _MultiScaleConv(nn.Module):
    """
    Parallel convolutions with kernel sizes 1, 3, 5 — captures local,
    medium, and global bit patterns simultaneously.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        branch = out_ch // 3
        extra  = out_ch - branch * 3

        self.k1 = nn.Sequential(
            nn.Conv1d(in_ch, branch + extra, kernel_size=1),
            nn.BatchNorm1d(branch + extra), nn.GELU(),
        )
        self.k3 = nn.Sequential(
            nn.Conv1d(in_ch, branch, kernel_size=3, padding=1),
            nn.BatchNorm1d(branch), nn.GELU(),
        )
        self.k5 = nn.Sequential(
            nn.Conv1d(in_ch, branch, kernel_size=5, padding=2),
            nn.BatchNorm1d(branch), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.k1(x), self.k3(x), self.k5(x)], dim=1)


class CNN(nn.Module):
    """
    CNN distinguisher for neural cryptanalysis.

    Key design decisions:
    - 3 input channels: C, C', and XOR-difference (C XOR C').
      The delta channel is the strongest differential signal — giving it
      explicitly as a channel lets the network learn from it directly.
    - Multi-scale stem: parallel k=1,3,5 convolutions to capture
      bit patterns at different scales simultaneously.
    - Residual blocks with dropout for depth + regularisation.
    - Global avg + max + std pooling for rich feature aggregation.
    - MLP head with skip connection from pooled features.
    """

    def __init__(self, input_dim: int, num_filters: int = 48, dropout: float = 0.1):
        super().__init__()

        assert input_dim % 2 == 0
        self.input_dim  = input_dim
        self.branch_dim = input_dim // 2

        # Multi-scale stem: 3 input channels -> num_filters
        self.stem = _MultiScaleConv(3, num_filters)

        # Stage 1
        self.stage1 = nn.Sequential(
            _ResBlock(num_filters, kernel_size=3, dropout=dropout),
            _ResBlock(num_filters, kernel_size=3, dropout=dropout),
        )

        # Widen to num_filters * 2
        f2 = num_filters * 2
        self.up1 = nn.Sequential(
            nn.Conv1d(num_filters, f2, kernel_size=1),
            nn.BatchNorm1d(f2), nn.GELU(),
        )

        # Stage 2
        self.stage2 = nn.Sequential(
            _ResBlock(f2, kernel_size=3, dropout=dropout),
            _ResBlock(f2, kernel_size=3, dropout=dropout),
        )

        # Widen to num_filters * 4
        f3 = num_filters * 4
        self.up2 = nn.Sequential(
            nn.Conv1d(f2, f3, kernel_size=1),
            nn.BatchNorm1d(f3), nn.GELU(),
        )

        # Stage 3
        self.stage3 = _ResBlock(f3, kernel_size=3, dropout=dropout)

        # Global pooling: avg + max + std -> f3 * 3
        pool_dim = f3 * 3

        # Classifier with skip from pooled features
        self.fc1   = nn.Linear(pool_dim, 256)
        self.ln1   = nn.LayerNorm(256)
        self.fc2   = nn.Linear(256, 64)
        self.ln2   = nn.LayerNorm(64)
        self.skip  = nn.Linear(pool_dim, 64, bias=False)
        self.drop  = nn.Dropout(dropout)
        self.act   = nn.GELU()
        self.out   = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c   = x[:, :self.branch_dim]          # bits(C)
        cp  = x[:, self.branch_dim:]          # bits(C')
        xor = (c - cp).abs()                  # approximate XOR on float bits

        # Stack 3 channels: (B, 3, branch_dim)
        x = torch.stack([c, cp, xor], dim=1)

        x = self.stem(x)
        x = self.stage1(x)
        x = self.up1(x)
        x = self.stage2(x)
        x = self.up2(x)
        x = self.stage3(x)

        # Global pooling
        avg  = x.mean(dim=2)
        mx   = x.max(dim=2).values
        std  = x.std(dim=2)
        feat = torch.cat([avg, mx, std], dim=1)   # (B, f3*3)

        # Classifier with skip
        skip = self.skip(feat)
        h    = self.act(self.ln1(self.fc1(feat)));  h = self.drop(h)
        h    = self.act(self.ln2(self.fc2(h)))
        h    = h + skip

        return self.out(h)
