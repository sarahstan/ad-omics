import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism with support for gene-gene interactions.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Linear projections for query, key, value
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.output_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        gene_regulatory_matrix: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of multi-head attention.

        Args:
            query: Tensor of shape [batch_size, seq_len_q, embed_dim]
            key: Tensor of shape [batch_size, seq_len_k, embed_dim]
            value: Tensor of shape [batch_size, seq_len_v, embed_dim]
            attn_mask: Optional tensor of shape [batch_size, seq_len] OR [batch_size, seq_len_q, seq_len_k]
                    Binary mask where 1 allows attention and 0 blocks it
            gene_regulatory_matrix: Optional tensor of shape [seq_len_q, seq_len_k]
                                Matrix encoding gene-gene regulatory interactions

        Returns:
            output: Tensor of shape [batch_size, seq_len_q, embed_dim]
            attention_weights: Tensor of shape [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]
        seq_len_v = value.shape[1]

        # Linear projections and reshape for multi-head attention
        q = (
            self.q_proj(query)
            .view(batch_size, seq_len_q, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(key)
            .view(batch_size, seq_len_k, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(value)
            .view(batch_size, seq_len_v, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)

        # Apply gene regulatory information if provided
        if gene_regulatory_matrix is not None:
            # Reshape for broadcasting across batch and heads
            gene_reg = gene_regulatory_matrix.unsqueeze(0).unsqueeze(0)
            scores = scores + gene_reg

        # Apply attention mask if provided
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                # attn_mask shape: [batch_size, seq_len] - padding mask
                # Convert to [batch_size, seq_len, seq_len] for self-attention
                batch_size_mask, seq_len = attn_mask.shape
                mask = attn_mask.unsqueeze(1) * attn_mask.unsqueeze(
                    2
                )  # [batch_size, seq_len, seq_len]
            elif attn_mask.dim() == 3:
                # attn_mask shape: [batch_size, seq_len_q, seq_len_k] - full attention mask
                mask = attn_mask
            else:
                raise ValueError(f"attn_mask should have 2 or 3 dimensions, got {attn_mask.dim()}")

            # Expand mask for broadcasting across heads: [batch_size, 1, seq_len_q, seq_len_k]
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Apply softmax and dropout with numerical stability
        # Handle the case where all values in a row might be -inf
        attention_weights = F.softmax(scores, dim=-1)

        # Replace NaN values with zeros (this can happen when all scores are -inf)
        attention_weights = torch.where(
            torch.isnan(attention_weights), torch.zeros_like(attention_weights), attention_weights
        )

        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        output = torch.matmul(attention_weights, v)

        # Reshape and project back to embed_dim
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.embed_dim)
        output = self.output_proj(output)

        return output, attention_weights
