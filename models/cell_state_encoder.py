import torch
import torch.nn as nn


class CellStateEncoder(nn.Module):
    """
    Encodes cell information (gene counts and cell type) into conditioned embeddings
    that preserve gene permutation equivariance.
    """

    def __init__(
        self,
        num_genes: int,  # Number of genes in reference list
        gene_embedding_dim: int,  # Dimension for gene embeddings
        num_cell_types: int,  # Number of cell types
        use_film: bool = True,  # Whether to use FiLM conditioning
        dropout: float = 0.1,  # Dropout rate
    ):
        super(CellStateEncoder, self).__init__()

        # Gene count embedding
        self.gene_embedding = nn.Linear(1, gene_embedding_dim)

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
        nn.init.normal_(self.gene_embedding.weight, mean=0.0, std=0.02)
        if self.gene_embedding.bias is not None:
            nn.init.zeros_(self.gene_embedding.bias)

        nn.init.normal_(self.cell_type_embedding.weight, mean=0.0, std=0.02)
        if self.cell_type_embedding.bias is not None:
            nn.init.zeros_(self.cell_type_embedding.bias)

    def forward(self, gene_counts: torch.Tensor, cell_type: torch.Tensor):
        """
        Encode cell state information

        Args:
            gene_counts: Tensor of shape [batch_size, num_genes]
                         Gene expression counts
            cell_type: Tensor of shape [batch_size, num_cell_types]
                       One-hot encoded cell type

        Returns:
            conditioned_embeddings: Tensor of shape [batch_size, num_genes, gene_embedding_dim]
                                   Gene embeddings conditioned on cell type
        """
        batch_size, num_genes = gene_counts.shape

        # Embed gene counts
        gene_counts = gene_counts.unsqueeze(-1)  # [batch_size, num_genes, 1]
        gene_embeddings = self.gene_embedding(
            gene_counts
        )  # [batch_size, num_genes, gene_embedding_dim]

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

        return self.dropout(conditioned_embeddings)
