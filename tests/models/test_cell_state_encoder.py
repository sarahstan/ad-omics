import pytest
import torch
from dataclasses import dataclass
from models.cell_state_encoder import CellStateEncoder


@dataclass
class CellStateEncoderParameters:
    """Parameters for the CellStateEncoder model."""

    def __init__(
        self,
        batch_size: int = 8,
        num_genes: int = 128,
        gene_embedding_dim: int = 16,
        num_cell_types: int = 12,
        use_film: bool = True,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize the parameters for the CellStateEncoder.
        Args:
            batch_size: Batch size for the model
            num_genes: Number of genes
            gene_embedding_dim: Dimension for gene embeddings
            num_cell_types: Number of cell types
            use_film: Whether to use FiLM conditioning
            dropout: Dropout rate
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_genes = num_genes
        self.gene_embedding_dim = gene_embedding_dim
        self.num_cell_types = num_cell_types
        self.use_film = use_film
        self.dropout = dropout


@pytest.fixture
def cse_params() -> CellStateEncoderParameters:
    """Fixture to create a batch size for testing."""
    return CellStateEncoderParameters()


@pytest.fixture
def cell_state_encoder(cse_params: CellStateEncoderParameters) -> CellStateEncoder:
    """Fixture to create a CellStateEncoder instance for testing."""
    return CellStateEncoder(
        num_genes=cse_params.num_genes,
        gene_embedding_dim=cse_params.gene_embedding_dim,
        num_cell_types=cse_params.num_cell_types,
        use_film=cse_params.use_film,
        dropout=cse_params.dropout,
    )


@pytest.fixture
def gene_counts(cse_params: CellStateEncoderParameters) -> torch.Tensor:
    """Fixture to create a tensor of gene counts."""
    return torch.randn(cse_params.batch_size, cse_params.num_genes).float()


@pytest.fixture
def cell_type(cse_params: CellStateEncoderParameters) -> torch.Tensor:
    """Fixture to create a tensor of cell types."""
    # Create a one-hot encoded tensor for cell types
    cell_type = torch.zeros(cse_params.batch_size, cse_params.num_cell_types).float()
    cell_type[
        torch.arange(cse_params.batch_size),
        torch.randint(0, cse_params.num_cell_types, (cse_params.batch_size,)),
    ] = 1.0
    return cell_type


def test_forward(
    cse_params: CellStateEncoderParameters,
    cell_state_encoder: CellStateEncoder,
    gene_counts: torch.Tensor,
    cell_type: torch.Tensor,
):
    """Test the forward method of CellStateEncoder."""
    output = cell_state_encoder(gene_counts, cell_type)

    assert output.shape == (
        cse_params.batch_size,
        cse_params.num_genes,
        cse_params.gene_embedding_dim,
    ), "Output shape mismatch"
    assert torch.all(torch.isfinite(output)), "Output contains non-finite values"


def test_forward_permutation_equivariance(
    cse_params: CellStateEncoderParameters,
    cell_state_encoder: CellStateEncoder,
    gene_counts: torch.Tensor,
    cell_type: torch.Tensor,
):
    """Test the permutation equivariance of the forward method."""
    # Set a deterministic seed
    torch.manual_seed(42)

    # Set model to evaluation mode
    cell_state_encoder.eval()

    # Create a permutation of gene counts
    permuted_indices = torch.randperm(cse_params.num_genes)
    permuted_gene_counts = gene_counts[:, permuted_indices]

    # Get the output for the original and permuted gene counts
    original_output = cell_state_encoder(gene_counts, cell_type)
    permuted_output = cell_state_encoder(permuted_gene_counts, cell_type)

    # Check if the output is equivariant to the permutation
    assert torch.allclose(
        original_output[:, permuted_indices, :],
        permuted_output,
        atol=1e-6,
    ), "Output is not permutation equivariant"
