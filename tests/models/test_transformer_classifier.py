import pytest
import torch
from models.torch.transformer_classifier import ADPredictionModel
from tests.utils import (
    create_permutation,
    create_permuted_data,
)


@pytest.fixture
def ad_prediction_model(
    model_params,
) -> ADPredictionModel:
    """Fixture to create an ADPredictionModel instance for testing."""
    return ADPredictionModel(
        # Model parameters
        batch_size=model_params.batch_size,
        # Cell State Encoder parameters
        num_genes=model_params.num_genes,
        num_cell_types=model_params.num_cell_types,
        embed_dim=model_params.embed_dim,
        use_film=model_params.use_film,
        max_number_genes=model_params.max_number_genes,
        cell_state_encoder_dropout=model_params.cell_state_encoder_dropout,
        # Transformer parameters
        num_heads=model_params.num_heads,
        ff_dim=model_params.ff_dim,
        num_layers=model_params.num_layers,
        transformer_dropout=model_params.transformer_dropout,
    )


def test_forward(
    model_params,
    ad_prediction_model: ADPredictionModel,
    gene_token_data: tuple,
    cell_type: torch.Tensor,
):
    """Test the forward method of ADPredictionModel."""
    gene_indices, gene_values, attention_mask = gene_token_data

    # Pass through the model
    logits, attention_weights = ad_prediction_model(
        gene_indices=gene_indices,
        gene_values=gene_values,
        cell_type_indices=cell_type,
        attention_mask=attention_mask,
    )

    # Check logits shape
    assert logits.shape == (model_params.batch_size,), "Logits shape mismatch"

    # Check attention weights shape: should be a list of tensors, one per layer
    assert (
        len(attention_weights) == model_params.num_layers
    ), "Incorrect number of attention weight layers"

    # Check that each attention weight is properly shaped
    for layer_idx, attn_weights in enumerate(attention_weights):
        expected_seq_len = model_params.max_seq_len

        expected_shape = (
            model_params.batch_size,
            model_params.num_heads,
            expected_seq_len,
            expected_seq_len,
        )

        error_str = f"Attention weights shape mismatch in layer {layer_idx}:"
        error_str += f" expected {expected_shape}, got {attn_weights.shape}"
        assert attn_weights.shape == expected_shape, error_str


def test_permutation_invariance(
    model_params,
    ad_prediction_model: ADPredictionModel,
    gene_token_data: tuple,
    cell_type: torch.Tensor,
):
    """
    Test the permutation invariance of ADPredictionModel.

    The model should produce the same final prediction (logits)
    regardless of the order of genes in the input.
    """
    # Set the model to evaluation mode for deterministic behavior
    ad_prediction_model.eval()

    # Test permutation invariance for one sample in the batch
    batch_idx = 0

    # Get the non-padded length for this sample
    seq_len = gene_token_data[2][batch_idx].sum().item()

    if seq_len <= 1:
        pytest.skip("Skipping test as sequence length is too short for meaningful permutation test")

    # Create a permutation of expressed genes for this sample
    torch.manual_seed(42)
    perm = create_permutation(seq_len)

    # Create original and permuted data
    original_indices, original_values, original_mask, permuted_indices, permuted_values = (
        create_permuted_data(gene_token_data, perm, seq_len, batch_idx)
    )

    # Get cell type for this sample
    sample_cell_type = cell_type[batch_idx : batch_idx + 1]

    # Process data through the model
    with torch.no_grad():
        # Original flow
        original_logits, original_attention = ad_prediction_model(
            gene_indices=original_indices,
            gene_values=original_values,
            cell_type_indices=sample_cell_type,
            attention_mask=original_mask,
        )

        # Permuted flow
        permuted_logits, permuted_attention = ad_prediction_model(
            gene_indices=permuted_indices,
            gene_values=permuted_values,
            cell_type_indices=sample_cell_type,
            attention_mask=original_mask,
        )

    # Check that the prediction (logits) remains unchanged
    # For permutation invariance, the output should be identical regardless of input order
    assert torch.allclose(
        original_logits, permuted_logits, atol=1e-5
    ), "ADPredictionModel is not permutation invariant - logits changed after gene permutation"
