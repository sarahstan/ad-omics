import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from .cell_state_encoder import CellStateEncoder
from .scrna_transformer import ScRNATransformer
from configs import (
    CellStateEncoderConfig,
    ScRNATransformerConfig,
)


class CellEncoderAndTransformerDimensionMismatchError(Exception):
    """Custom exception for dimension mismatch between cell encoder and transformer."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class ADPredictionModel(nn.Module):
    """
    Complete model for AD prediction from scRNA-seq data.
    Combines CellStateEncoder and ScRNATransformer with full permutation equivariance.
    """

    def __init__(
        self,
        cell_state_encoder_config: CellStateEncoderConfig,
        scrna_transformer_config: ScRNATransformerConfig,
    ):
        super(ADPredictionModel, self).__init__()

        # Cell state encoder (passed in as initialized module)
        self.cell_state_encoder = CellStateEncoder(cell_state_encoder_config)

        # Transformer with permutation equivariance
        self.transformer = ScRNATransformer(scrna_transformer_config)

        self._validate_configs()

    def _validate_configs(self):
        """Verify the two configurations have the same internal dimension."""
        cse_dim = self.cell_state_encoder.config.gene_embedding_dim
        transformer_dim = self.transformer.config.embed_dim
        error_str = "Expected cell encoder and transformer to have the same internal dimension. "
        error_str += f"Encoder gene_embedding_dim = {cse_dim}, "
        error_str += f"Transformer embed_dim = {transformer_dim}"
        raise CellEncoderAndTransformerDimensionMismatchError(error_str)

    def forward(
        self,
        gene_indices: torch.Tensor,
        gene_values: torch.Tensor,
        cell_type_indices: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        gene_regulatory_matrix: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass of the complete AD prediction model with permutation invariance.

        Args:
            gene_indices: Tensor of shape [batch_size, max_seq_len]
                          Indices of expressed genes
            gene_values: Tensor of shape [batch_size, max_seq_len]
                         Expression values for each gene
            cell_type_indices: Tensor of shape [batch_size]
                               Indices of the cell type
            attention_mask: Optional tensor of shape [batch_size, max_seq_len]
                           Attention mask where 1 means token should be attended
                           to and 0 means it should be ignored
            gene_regulatory_matrix: Optional tensor of shape [max_seq_len, max_seq_len]
                                    Matrix encoding gene-gene regulatory interactions

        Returns:
            logits: Prediction logits for AD classification
            attention_weights: List of attention weights from all transformer layers
        """
        # Encode cell state
        cell_embeddings = self.cell_state_encoder(
            gene_indices=gene_indices,
            gene_values=gene_values,
            cell_type_indices=cell_type_indices,
            attention_mask=attention_mask,
        )

        # Pass through transformer
        logits, attention_weights = self.transformer(
            x=cell_embeddings,
            attention_mask=attention_mask,
            gene_regulatory_matrix=gene_regulatory_matrix,
        )

        return logits, attention_weights
