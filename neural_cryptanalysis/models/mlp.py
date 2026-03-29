import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    MLP distinguisher for neural cryptanalysis.

    Improvements:
    - Wider first layer (1024) to capture more bit combinations
    - Two residual skip connections (input->mid, input->out)
    - LayerNorm instead of BatchNorm (more stable on small batches)
    - GELU activations throughout
    - Deeper path: 1024 -> 512 -> 256 -> 128 -> 64
    """

    def __init__(self, input_dim: int, dropout: float = 0.15):
        super().__init__()
        self.input_dim = input_dim

        self.fc1 = nn.Linear(input_dim, 1024)
        self.ln1 = nn.LayerNorm(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.ln2 = nn.LayerNorm(512)

        self.fc3 = nn.Linear(512, 256)
        self.ln3 = nn.LayerNorm(256)

        self.fc4 = nn.Linear(256, 128)
        self.ln4 = nn.LayerNorm(128)

        self.fc5 = nn.Linear(128, 64)
        self.ln5 = nn.LayerNorm(64)

        # Two skip projections
        self.skip1 = nn.Linear(input_dim, 512, bias=False)   # input -> after fc2
        self.skip2 = nn.Linear(input_dim, 64,  bias=False)   # input -> after fc5

        self.drop = nn.Dropout(dropout)
        self.act  = nn.GELU()

        self.out = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.skip1(x)
        s2 = self.skip2(x)

        h = self.act(self.ln1(self.fc1(x)));  h = self.drop(h)
        h = self.act(self.ln2(self.fc2(h)))
        h = h + s1;                           h = self.drop(h)
        h = self.act(self.ln3(self.fc3(h)));  h = self.drop(h)
        h = self.act(self.ln4(self.fc4(h)));  h = self.drop(h)
        h = self.act(self.ln5(self.fc5(h)))
        h = h + s2

        return self.out(h)
