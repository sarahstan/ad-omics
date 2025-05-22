from dataclasses import dataclass


@dataclass
class CellStateEncoderConfig:
    """
    Configuration class for CellStateEncoder.
    """

    num_genes_total: int = 15000
    gene_embedding_dim: int = 16
    num_cell_types: int = 11
    use_film: bool = True
    dropout: float = 0.1
