import pytest
import torch
from models.transformer.classifier import (
    ADPredictionModel,
    CellEncoderAndTransformerDimensionMismatchError,
)
from configs import CellStateEncoderConfig, ScRNATransformerConfig
from tests.utils import (
    create_permutation,
    create_permuted_data,
)


@pytest.fixture
def ad_prediction_model(
    cell_state_encoder_config: CellStateEncoderConfig,
    scrna_transformer_config: ScRNATransformerConfig,
) -> ADPredictionModel:
    """Fixture to create an ADPredictionModel instance for testing."""
    return ADPredictionModel(
        cell_state_encoder_config=cell_state_encoder_config,
        scrna_transformer_config=scrna_transformer_config,
    )


def test_forward(
    ad_prediction_model: ADPredictionModel,
    cell_state_encoder_config: CellStateEncoderConfig,
    scrna_transformer_config: ScRNATransformerConfig,
    gene_token_data: tuple,
    cell_type: torch.Tensor,
    batch_size: int,
    num_genes_per_cell_max: int,
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
    assert logits.shape == (batch_size,), "Logits shape mismatch"

    # Check attention weights shape: should be a list of tensors, one per layer
    assert (
        len(attention_weights) == scrna_transformer_config.num_layers
    ), "Incorrect number of attention weight layers"

    # Check that each attention weight is properly shaped
    for layer_idx, attn_weights in enumerate(attention_weights):
        expected_seq_len = num_genes_per_cell_max

        expected_shape = (
            batch_size,
            scrna_transformer_config.num_heads,
            expected_seq_len,
            expected_seq_len,
        )

        error_str = f"Attention weights shape mismatch in layer {layer_idx}:"
        error_str += f" expected {expected_shape}, got {attn_weights.shape}"
        assert attn_weights.shape == expected_shape, error_str


def test_permutation_invariance(
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


import copy


def test_validation_fails_when_embed_dims_mismatch(
    cell_state_encoder_config: CellStateEncoderConfig,
    scrna_transformer_config: ScRNATransformerConfig,
):
    bad_encoder_config = copy.deepcopy(cell_state_encoder_config)
    bad_transformer_config = copy.deepcopy(scrna_transformer_config)

    bad_encoder_config.gene_embedding_dim = 5
    bad_transformer_config.embed_dim = 4

    with pytest.raises(CellEncoderAndTransformerDimensionMismatchError) as exc_info:
        _ = ADPredictionModel(
            cell_state_encoder_config=bad_encoder_config,
            scrna_transformer_config=bad_transformer_config,
        )
    correct_str = "Expected cell encoder and transformer to have the same internal dimension"
    condition = correct_str in str(exc_info.value)
    error_str = "Error message does not contain expected string"
    assert condition, error_str


def test_validation_passes_when_embed_dims_match(
    cell_state_encoder_config: CellStateEncoderConfig,
    scrna_transformer_config: ScRNATransformerConfig,
):
    """Test that matching dimensions don't raise an error"""
    try:
        _ = ADPredictionModel(
            cell_state_encoder_config=cell_state_encoder_config,
            scrna_transformer_config=scrna_transformer_config,
        )
        # If we get here, the test passes
    except CellEncoderAndTransformerDimensionMismatchError as e:
        pytest.fail(f"Expected no exception for matching dimensions, but got: {e}")
    except Exception as e:
        pytest.fail(f"Unexpected exception type: {type(e).__name__}: {e}")
