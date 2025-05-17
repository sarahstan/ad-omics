from dataclasses import dataclass
import torch
from typing import Optional


@dataclass
class VectorData:
    """
    Represents gene expression data in vector format.

    For single samples:
      - gene_vector: [num_genes]
      - cell_type: [num_cell_types]
      - label: scalar

    For batches:
      - gene_vector: [batch_size, num_genes]
      - cell_type: [batch_size, num_cell_types]
      - label: [batch_size]
    """

    gene_vector: torch.Tensor
    cell_type: torch.Tensor
    label: torch.Tensor


@dataclass
class TokenData:
    """
    Represents gene expression data in token format.

    For single samples:
      - gene_indices: [num_expressed_genes]
      - gene_values: [num_expressed_genes]
      - cell_type: [num_cell_types]
      - label: scalar
      - attention_mask: None (not used for individual samples)

    For batches:
      - gene_indices: [batch_size, max_seq_len]
      - gene_values: [batch_size, max_seq_len]
      - cell_type: [batch_size, num_cell_types]
      - label: [batch_size]
      - attention_mask: [batch_size, max_seq_len]
    """

    gene_indices: torch.Tensor
    gene_counts: torch.Tensor
    cell_type: torch.Tensor
    label: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None
