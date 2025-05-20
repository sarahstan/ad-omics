import pytest
import torch
from dataclasses import dataclass
from models.cell_state_encoder import CellStateEncoder
from tests.utils import create_permutation, create_inverse_permutation, create_permuted_data


@dataclass
class CellStateEncoderParameters:
    """Parameters for the CellStateEncoder model."""

    def __init__(
        self,
        batch_size: int = 8,
        num_genes: int = 15000,  # Total vocabulary size
        max_seq_len: int = 500,  # Maximum sequence length (expressed genes)
        gene_embedding_dim: int = 16,
        num_cell_types: int = 12,
        use_film: bool = True,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the parameters for CellStateEncoder tests."""
        super().__init__()
        # Set attributes from the init parameters
        for key, value in locals().items():
            if key != "self":
                setattr(self, key, value)


@pytest.fixture
def cse_params(scdata) -> CellStateEncoderParameters:
    """Fixture to create parameters for testing."""
    params = CellStateEncoderParameters()
    params.num_cell_types = len(scdata.cell_types)
    return params


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
def gene_token_data(cse_params: CellStateEncoderParameters, generate_gene_data) -> tuple:
    """Fixture to create token representation of gene data."""
    return generate_gene_data(
        batch_size=cse_params.batch_size,
        num_genes=cse_params.num_genes,
        max_seq_len=cse_params.max_seq_len,
    )


@pytest.fixture
def cell_type(cse_params: CellStateEncoderParameters) -> torch.Tensor:
    """Fixture to create a tensor of cell types."""
    # Create an integer tensor representing cell types
    torch.manual_seed(42)  # For reproducibility
    cell_type = torch.randint(0, cse_params.num_cell_types, (cse_params.batch_size,))
    return cell_type


def test_forward(
    cse_params: CellStateEncoderParameters,
    cell_state_encoder: CellStateEncoder,
    gene_token_data: tuple,
    cell_type: torch.Tensor,
):
    """Test the forward method of CellStateEncoder."""
    gene_indices, gene_values, attention_mask = gene_token_data

    output = cell_state_encoder(gene_indices, gene_values, cell_type, attention_mask)

    assert output.shape == (
        cse_params.batch_size,
        cse_params.max_seq_len,
        cse_params.gene_embedding_dim,
    ), "Output shape mismatch"
    assert torch.all(torch.isfinite(output)), "Output contains non-finite values"


def test_forward_permutation_equivariance(
    cse_params: CellStateEncoderParameters,
    cell_state_encoder: CellStateEncoder,
    gene_token_data: tuple,
    cell_type: torch.Tensor,
):
    """Test the permutation equivariance of the forward method with token representation."""
    # Set a deterministic seed
    torch.manual_seed(42)

    # Set model to evaluation mode
    cell_state_encoder.eval()

    gene_indices, gene_values, attention_mask = gene_token_data

    # Test permutation equivariance for each sample in the batch
    for batch_idx in range(cse_params.batch_size):
        # Get the non-padded length for this sample
        seq_len = attention_mask[batch_idx].sum().item()

        if seq_len <= 1:  # Only test if we have at least 2 expressed genes
            continue

        # Create a permutation of expressed genes for this sample
        perm = create_permutation(seq_len)

        # Create original and permuted data
        original_indices, original_values, original_mask, permuted_indices, permuted_values = (
            create_permuted_data(gene_token_data, perm, seq_len, batch_idx)
        )

        sample_cell_type = cell_type[batch_idx : batch_idx + 1]

        # Get the output for the original and permuted data
        with torch.no_grad():
            original_output = cell_state_encoder(
                original_indices, original_values, sample_cell_type, original_mask
            )

            permuted_output = cell_state_encoder(
                permuted_indices, permuted_values, sample_cell_type, original_mask
            )

        # Extract the non-padded portion
        orig_emb = original_output[0, :seq_len]
        perm_emb = permuted_output[0, :seq_len]

        # Create mapping from permuted positions back to original positions
        inverse_perm = create_inverse_permutation(perm)

        # Reorder the permuted embeddings to match the original order
        reordered_perm_emb = perm_emb[inverse_perm]

        # Check if the embeddings are equivariant to the permutation
        assert torch.allclose(
            orig_emb,
            reordered_perm_emb,
            atol=1e-6,
        ), f"Output for batch {batch_idx} is not permutation equivariant"
