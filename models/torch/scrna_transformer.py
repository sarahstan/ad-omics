import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from .transformer_encoder import TransformerEncoderLayer
from configs import ScRNATransformerConfig


class ScRNATransformer(nn.Module):
    """
    Transformer-based model for scRNA-seq data analysis to predict AD state.
    Uses gene embeddings from CellStateEncoder with full permutation equivariance.
    """

    def __init__(
        self,
        config: ScRNATransformerConfig,
    ):
        super(ScRNATransformer, self).__init__()

        self.config = config

        # Transformer encoder layers
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    self.config.embed_dim,
                    self.config.num_heads,
                    self.config.ff_dim,
                    self.config.dropout,
                )
                for _ in range(self.config.num_layers)
            ]
        )

        # Layer norm for final output
        self.norm = nn.LayerNorm(self.config.embed_dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.config.embed_dim, self.config.embed_dim),
            nn.Tanh(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.embed_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        gene_regulatory_matrix: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass of the ScRNA transformer.

        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim] from CellStateEncoder
            attention_mask: Optional tensor of shape [batch_size, seq_len]
                            Attention mask where 1 means token should be attended to
                            and 0 means it should be ignored
            gene_regulatory_matrix: Optional tensor of shape [seq_len, seq_len]
                                   Matrix encoding gene-gene regulatory interactions

        Returns:
            logits: Prediction logits for AD classification
            attention_weights: List of attention weights from all layers
        """
        # Pass through transformer layers
        attention_weights_list = []
        for layer in self.layers:
            x, attention_weights = layer(
                x, attn_mask=attention_mask, gene_regulatory_matrix=gene_regulatory_matrix
            )
            attention_weights_list.append(attention_weights)

        # Apply final layer norm
        x = self.norm(x)

        # Use mean pooling across sequence length for permutation invariance
        if attention_mask is not None:
            # Use attention mask for proper mean calculation
            mask_expanded = attention_mask.unsqueeze(-1).float()
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            # Simple mean pooling if no mask
            x = x.mean(dim=1)

        # Classification
        logits = self.classifier(x).squeeze(-1)

        return logits, attention_weights_list
