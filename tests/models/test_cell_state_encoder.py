import pytest
import torch
from dataclasses import dataclass
from models.cell_state_encoder import CellStateEncoder


@dataclass
class CellStateEncoderParameters:
    """
    Parameters for the CellStateEncoder model.
    Made up for the purpose of testing.
    """

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
        """
        Initialize the parameters for the CellStateEncoder.
        Args:
            batch_size: Batch size for the model
            num_genes: Total vocabulary of genes
            max_seq_len: Maximum sequence length (expressed genes per cell)
            gene_embedding_dim: Dimension for gene embeddings
            num_cell_types: Number of cell types
            use_film: Whether to use FiLM conditioning
            dropout: Dropout rate
        """
        super().__init__()
        # Set attributes from the init parameters
        for key, value in locals().items():
            if key != "self":
                setattr(self, key, value)


@pytest.fixture
def cse_params() -> CellStateEncoderParameters:
    """Fixture to create parameters for testing."""
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
def gene_token_data(cse_params: CellStateEncoderParameters) -> tuple:
    """Fixture to create token representation of gene data."""
    # Generate unique random gene indices for each sample in the batch
    gene_indices = []
    gene_values = []

    for _ in range(cse_params.batch_size):
        # Sample a random number of expressed genes (between 100 and max_seq_len)
        num_expressed = torch.randint(100, cse_params.max_seq_len, (1,)).item()

        # Sample random gene indices (without replacement)
        indices = torch.randperm(cse_params.num_genes)[:num_expressed]
        gene_indices.append(indices)

        # Generate random expression values for each gene
        values = torch.abs(torch.randn(num_expressed))
        gene_values.append(values)

    # Create attention mask and pad to max_seq_len
    padded_indices = torch.zeros(cse_params.batch_size, cse_params.max_seq_len, dtype=torch.long)
    padded_values = torch.zeros(cse_params.batch_size, cse_params.max_seq_len, dtype=torch.float)
    attention_mask = torch.zeros(cse_params.batch_size, cse_params.max_seq_len, dtype=torch.bool)

    for i, (indices, values) in enumerate(zip(gene_indices, gene_values)):
        seq_len = indices.size(0)
        padded_indices[i, :seq_len] = indices
        padded_values[i, :seq_len] = values
        attention_mask[i, :seq_len] = True

    return padded_indices, padded_values, attention_mask


@pytest.fixture
def cell_type(cse_params: CellStateEncoderParameters) -> torch.Tensor:
    """Fixture to create a tensor of cell types."""
    # Create a one-hot encoded tensor for cell types
    cell_type = torch.zeros(cse_params.batch_size, cse_params.num_cell_types)
    cell_type[
        torch.arange(cse_params.batch_size),
        torch.randint(0, cse_params.num_cell_types, (cse_params.batch_size,)),
    ] = 1.0
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

        if seq_len > 1:  # Only test if we have at least 2 expressed genes
            # Create a permutation of expressed genes for this sample
            perm = torch.randperm(seq_len)

            # Apply permutation to this sample's data
            sample_indices = gene_indices[batch_idx, :seq_len][perm]
            sample_values = gene_values[batch_idx, :seq_len][perm]

            # Create input tensors with just this single sample
            original_indices = gene_indices[batch_idx : batch_idx + 1].clone()
            original_values = gene_values[batch_idx : batch_idx + 1].clone()
            original_mask = attention_mask[batch_idx : batch_idx + 1].clone()

            permuted_indices = original_indices.clone()
            permuted_values = original_values.clone()
            permuted_indices[0, :seq_len] = sample_indices
            permuted_values[0, :seq_len] = sample_values

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

            # The permuted embeddings should match the original embeddings when reordered
            # Create mapping from permuted positions back to original positions
            inverse_perm = torch.zeros(seq_len, dtype=torch.long)
            for i, p in enumerate(perm):
                inverse_perm[p] = i

            reordered_perm_emb = perm_emb[inverse_perm]

            # Check if the embeddings are equivariant to the permutation
            assert torch.allclose(
                orig_emb,
                reordered_perm_emb,
                atol=1e-6,
            ), f"Output for batch {batch_idx} is not permutation equivariant"
