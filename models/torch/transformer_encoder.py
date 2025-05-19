import torch
import torch.nn as nn
from typing import Optional, Tuple
from .multi_head_attention import MultiHeadAttention
from .feed_forward import FeedForward


class TransformerEncoderLayer(nn.Module):
    """
    Single layer of the transformer encoder with support for gene regulatory networks.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
    ):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.feed_forward = FeedForward(embed_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        gene_regulatory_matrix: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the transformer encoder layer.

        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim]
            attn_mask: Optional tensor for masking attention
            gene_regulatory_matrix: Optional matrix of gene-gene regulatory interactions

        Returns:
            output: Processed tensor of shape [batch_size, seq_len, embed_dim]
            attention_weights: Attention weights for visualization
        """
        # Self-attention with residual connection and layer norm
        attn_output, attention_weights = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_mask=attn_mask,
            gene_regulatory_matrix=gene_regulatory_matrix,
        )
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x, attention_weights
