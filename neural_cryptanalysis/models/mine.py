import torch
import torch.nn as nn


class MINE(nn.Module):
    """
    Mutual Information Neural Estimator (MINE) adapted as a binary distinguisher.

    Reference: Belghazi et al., ICML 2018 — "MINE: Mutual Information Neural Estimation"

    Key improvements over a naive MLP:
    - Splits input into (C, C') and computes XOR difference explicitly,
      then feeds [C || C' || ΔC] to the statistics network — giving the
      model the differential signal directly (same insight as the CNN).
    - Deep statistics network with LayerNorm + GELU + skip connections,
      matching the MLP architecture that achieves ~99% accuracy.
    - Wider hidden layers (512) to capture complex bit interactions.

    Architecture:
        Input : concat(C, C') — shape (2 * block_bits,)
        Augment: append ΔC = C XOR C' → shape (3 * block_bits,)
        T-net  : Linear(512)->LN->GELU x4 + 2 skip connections -> Linear(1)->Sigmoid
        Output : sigmoid score in [0, 1]
    """

    def __init__(self, input_dim: int, hidden_dim: int = 512):
        super().__init__()

        assert input_dim % 2 == 0
        self.input_dim  = input_dim
        self.branch_dim = input_dim // 2

        # Augmented input: C || C' || ΔC  →  3 * branch_dim
        aug_dim = input_dim + self.branch_dim   # 3 * branch_dim

        self.fc1 = nn.Linear(aug_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.ln3 = nn.LayerNorm(hidden_dim // 2)

        self.fc4 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.ln4 = nn.LayerNorm(hidden_dim // 4)

        # Skip connections
        self.skip1 = nn.Linear(aug_dim, hidden_dim,      bias=False)  # input → after fc2
        self.skip2 = nn.Linear(aug_dim, hidden_dim // 4, bias=False)  # input → after fc4

        self.drop = nn.Dropout(0.1)
        self.act  = nn.GELU()

        self.out = nn.Sequential(
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split and compute differential channel
        c   = x[:, :self.branch_dim]
        cp  = x[:, self.branch_dim:]
        xor = torch.clamp((c - cp).abs(), 0, 1)   # binary XOR approximation

        # Augmented input: [C || C' || ΔC]
        x_aug = torch.cat([c, cp, xor], dim=1)

        s1 = self.skip1(x_aug)
        s2 = self.skip2(x_aug)

        h = self.act(self.ln1(self.fc1(x_aug))); h = self.drop(h)
        h = self.act(self.ln2(self.fc2(h)))
        h = h + s1;                               h = self.drop(h)
        h = self.act(self.ln3(self.fc3(h)));      h = self.drop(h)
        h = self.act(self.ln4(self.fc4(h)))
        h = h + s2

        return self.out(h)
