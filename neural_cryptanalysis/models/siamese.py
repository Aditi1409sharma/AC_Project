import torch
import torch.nn as nn


class SiameseNet(nn.Module):
    """
    Siamese (twin) network for neural cryptanalysis.

    Takes a ciphertext pair (C, C') as two separate branches.
    Each branch encodes its input independently with shared weights,
    then the embeddings are combined and passed to a classifier.

    Input: flat vector of length 2 * branch_dim
           first half  = bits(C)
           second half = bits(C')

    The shared encoder learns what structure to look for in a single
    ciphertext, and the merge layer learns how the two relate.
    """

    def __init__(self, branch_dim: int, embed_dim: int = 64):
        """
        Args:
            branch_dim : number of bits per ciphertext (= block_bits)
            embed_dim  : size of each branch embedding
        """
        super().__init__()

        self.branch_dim = branch_dim

        # Shared encoder — same weights applied to C and C'
        self.encoder = nn.Sequential(
            nn.Linear(branch_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
            nn.ReLU(),
        )

        # Merge: concatenate embeddings + element-wise difference
        # merged dim = embed_dim * 2 + embed_dim = embed_dim * 3
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 3, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split input into two halves
        c  = x[:, :self.branch_dim]
        cp = x[:, self.branch_dim:]

        # Encode each branch with shared weights
        emb_c  = self.encoder(c)
        emb_cp = self.encoder(cp)

        # Merge: concat + absolute difference
        diff   = torch.abs(emb_c - emb_cp)
        merged = torch.cat([emb_c, emb_cp, diff], dim=1)

        return self.classifier(merged)
