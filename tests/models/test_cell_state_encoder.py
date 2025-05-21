import pytest
import torch
from models.torch.cell_state_encoder import CellStateEncoder
from tests.utils import create_permutation, create_inverse_permutation, create_permuted_data


def test_forward(
    model_params,
    cell_state_encoder: CellStateEncoder,
    gene_token_data: tuple,
    cell_type: torch.Tensor,
):
    """Test the forward method of CellStateEncoder."""
    gene_indices, gene_values, attention_mask = gene_token_data

    output = cell_state_encoder(gene_indices, gene_values, cell_type, attention_mask)

    assert output.shape == (
        model_params.batch_size,
        model_params.num_genes_per_cell_max,
        model_params.embed_dim,
    ), "Output shape mismatch"
    assert torch.all(torch.isfinite(output)), "Output contains non-finite values"


def test_forward_permutation_equivariance(
    model_params,
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
    for batch_idx in range(model_params.batch_size):
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
