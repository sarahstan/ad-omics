from typing import Tuple
from dataclasses import dataclass
import torch
from typing import Optional

VectorDataTensorType = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
TokenDataTensorType = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]
]


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

    @staticmethod
    def collate(batch) -> VectorDataTensorType:
        """
        Collate function to handle batching of data.
        """
        gene_vector = torch.stack([item.gene_vector for item in batch])
        cell_type = torch.stack([item.cell_type for item in batch])
        label = torch.tensor([item.label for item in batch])

        return gene_vector, cell_type, label


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

    @staticmethod
    def collate(batch) -> TokenDataTensorType:
        """
        Collate function to handle batching of data.
        """
        gene_indices = torch.nn.utils.rnn.pad_sequence(
            [item.gene_indices for item in batch], batch_first=True
        )
        gene_counts = torch.nn.utils.rnn.pad_sequence(
            [item.gene_counts for item in batch], batch_first=True
        )
        cell_type = torch.stack([item.cell_type for item in batch])
        label = torch.tensor([item.label for item in batch])
        attention_mask = (gene_indices != 0).float()

        return gene_indices, gene_counts, cell_type, label, attention_mask
