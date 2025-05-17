import torch
import torch.nn as nn
from typing import Optional


class CellStateEncoder(nn.Module):
    """
    Encodes cell information (gene IDs, counts, and cell type) into conditioned embeddings
    that preserve gene permutation equivariance.

    Works with sparse token representation where only expressed genes are included.
    """

    def __init__(
        self,
        num_genes: int,  # Total vocabulary of genes (e.g., 15,000)
        gene_embedding_dim: int,  # Dimension for gene embeddings
        num_cell_types: int,  # Number of cell types
        use_film: bool = True,  # Whether to use FiLM conditioning
        dropout: float = 0.1,  # Dropout rate
    ):
        super(CellStateEncoder, self).__init__()

        # Gene identity embedding (like word embeddings in NLP)
        self.gene_id_embedding = nn.Embedding(num_genes, gene_embedding_dim)

        # Gene count embedding
        self.count_embedding = nn.Linear(1, gene_embedding_dim)

        # Cell type embedding
        self.cell_type_embedding = nn.Linear(num_cell_types, gene_embedding_dim)

        # FiLM conditioning
        self.use_film = use_film
        if use_film:
            self.gamma_projection = nn.Sequential(
                nn.Linear(gene_embedding_dim, gene_embedding_dim),
                nn.ReLU(),
                nn.Linear(gene_embedding_dim, gene_embedding_dim),
            )

            self.beta_projection = nn.Sequential(
                nn.Linear(gene_embedding_dim, gene_embedding_dim),
                nn.ReLU(),
                nn.Linear(gene_embedding_dim, gene_embedding_dim),
            )

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize embeddings and projections with small values
        nn.init.normal_(self.gene_id_embedding.weight, mean=0.0, std=0.02)

        nn.init.normal_(self.count_embedding.weight, mean=0.0, std=0.02)
        if self.count_embedding.bias is not None:
            nn.init.zeros_(self.count_embedding.bias)

        nn.init.normal_(self.cell_type_embedding.weight, mean=0.0, std=0.02)
        if self.cell_type_embedding.bias is not None:
            nn.init.zeros_(self.cell_type_embedding.bias)

    def forward(
        self,
        gene_indices: torch.Tensor,
        gene_values: torch.Tensor,
        cell_type: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Encode cell state information using sparse token representation

        Args:
            gene_indices: Tensor of shape [batch_size, max_seq_len]
                         Indices of expressed genes (integers from 0 to num_genes-1)
            gene_values: Tensor of shape [batch_size, max_seq_len]
                         Expression counts for each gene
            cell_type: Tensor of shape [batch_size, num_cell_types]
                       One-hot encoded cell type
            attention_mask: Optional tensor of shape [batch_size, max_seq_len]
                           Boolean mask for padding (1 for real tokens, 0 for padding)

        Returns:
            conditioned_embeddings: Tensor of shape [batch_size, max_seq_len, gene_embedding_dim]
                                   Gene embeddings conditioned on cell type
        """
        batch_size, max_seq_len = gene_indices.shape

        # Embed gene identities
        gene_id_embeddings = self.gene_id_embedding(
            gene_indices
        )  # [batch_size, max_seq_len, gene_embedding_dim]

        # Embed gene counts
        gene_values = gene_values.unsqueeze(-1)  # [batch_size, max_seq_len, 1]
        count_embeddings = self.count_embedding(
            gene_values
        )  # [batch_size, max_seq_len, gene_embedding_dim]

        # Combine gene identity and count embeddings
        gene_embeddings = (
            gene_id_embeddings + count_embeddings
        )  # [batch_size, max_seq_len, gene_embedding_dim]

        # Embed cell type
        cell_embedding = self.cell_type_embedding(cell_type)  # [batch_size, gene_embedding_dim]

        if self.use_film:
            # Generate FiLM conditioning parameters
            gamma = self.gamma_projection(cell_embedding)  # [batch_size, gene_embedding_dim]
            beta = self.beta_projection(cell_embedding)  # [batch_size, gene_embedding_dim]

            # Reshape for broadcasting
            gamma = gamma.unsqueeze(1)  # [batch_size, 1, gene_embedding_dim]
            beta = beta.unsqueeze(1)  # [batch_size, 1, gene_embedding_dim]

            # Apply FiLM conditioning
            conditioned_embeddings = gene_embeddings * gamma + beta
        else:
            # Simple addition of cell embedding to each gene embedding
            cell_embedding = cell_embedding.unsqueeze(1)  # [batch_size, 1, gene_embedding_dim]
            conditioned_embeddings = gene_embeddings + cell_embedding

        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask for broadcasting across embedding dimension
            mask = attention_mask.unsqueeze(-1)  # [batch_size, max_seq_len, 1]
            conditioned_embeddings = conditioned_embeddings * mask

        return self.dropout(conditioned_embeddings)
