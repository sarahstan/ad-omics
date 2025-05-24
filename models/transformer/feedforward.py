import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network used in transformer.
    """

    def __init__(
        self,
        embed_dim: int,
        ff_dim: int,
        dropout: float = 0.1,
    ):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the feed-forward network.

        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim]

        Returns:
            Output tensor of shape [batch_size, seq_len, embed_dim]
        """
        return self.dropout(self.linear2(F.gelu(self.linear1(x))))
