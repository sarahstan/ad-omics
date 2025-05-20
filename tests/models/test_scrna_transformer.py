import pytest
import torch
from models.cell_state_encoder import CellStateEncoder
from models.torch.scrna_transformer import ScRNATransformer
from tests.utils import (
    create_permutation,
    create_inverse_permutation,
    create_permuted_data,
    get_expected_attention_shape,
    check_cls_token_attention,
    check_sequence_attention,
)


@pytest.fixture
def scrna_transformer(model_params) -> ScRNATransformer:
    """Fixture to create a ScRNATransformer instance for testing."""
    return ScRNATransformer(
        embed_dim=model_params.embed_dim,
        num_heads=model_params.num_heads,
        ff_dim=model_params.ff_dim,
        num_layers=model_params.num_layers,
        max_seq_len=model_params.max_seq_len,
        dropout=model_params.dropout,
        use_cls_token=model_params.use_cls_token,
    )


def test_forward(
    model_params,
    cell_state_encoder: CellStateEncoder,
    scrna_transformer: ScRNATransformer,
    gene_token_data: tuple,
    cell_type: torch.Tensor,
):
    """Test the forward method of ScRNATransformer."""
    gene_indices, gene_values, attention_mask = gene_token_data

    # Get embeddings from CellStateEncoder
    gene_embeddings = cell_state_encoder(gene_indices, gene_values, cell_type, attention_mask)

    # Convert boolean mask to float mask
    float_attention_mask = attention_mask.float()

    # Pass through transformer
    logits, attention_weights = scrna_transformer(gene_embeddings, float_attention_mask)

    # Check logits shape
    assert logits.shape == (model_params.batch_size,), "Logits shape mismatch"

    # Check attention weights shape: should be a list of tensors, one per layer
    assert (
        len(attention_weights) == model_params.num_layers
    ), "Incorrect number of attention weight layers"

    # Get expected attention shape
    expected_shape = get_expected_attention_shape(
        model_params.batch_size,
        model_params.num_heads,
        model_params.max_seq_len,
        model_params.use_cls_token,
    )

    # Check shape of each attention weight tensor
    for layer_idx, attn_weights in enumerate(attention_weights):
        error_str = f"Attention weights shape mismatch in layer {layer_idx}:"
        error_str += f" expected {expected_shape}, got {attn_weights.shape}"
        assert attn_weights.shape == expected_shape, error_str


def test_transformer_permutation_equivariance(
    model_params,
    cell_state_encoder: CellStateEncoder,
    scrna_transformer: ScRNATransformer,
    gene_token_data: tuple,
    cell_type: torch.Tensor,
):
    """
    Test the permutation equivariance of the ScRNATransformer.

    The test verifies that permuting the order of genes in the input
    should maintain the same relative attention patterns (after accounting
    for the permutation).
    """
    # Set models to evaluation mode
    cell_state_encoder.eval()
    scrna_transformer.eval()

    # Convert boolean mask to float mask
    attention_mask = gene_token_data[2]

    # Test permutation equivariance for one sample in the batch
    batch_idx = 0

    # Get the non-padded length for this sample
    seq_len = attention_mask[batch_idx].sum().item()

    if seq_len <= 1:
        pytest.skip("Skipping test as sequence length is too short for meaningful permutation test")

    # Create a permutation of expressed genes for this sample
    torch.manual_seed(42)
    perm = create_permutation(seq_len)

    # Create original and permuted data
    original_indices, original_values, original_mask, permuted_indices, permuted_values = (
        create_permuted_data(gene_token_data, perm, seq_len, batch_idx)
    )

    # Create float mask for transformer
    original_float_mask = original_mask.float()

    # Get cell type for this sample
    sample_cell_type = cell_type[batch_idx : batch_idx + 1]

    # Create inverse permutation mapping
    inverse_perm = create_inverse_permutation(perm)

    # Process data through the models
    with torch.no_grad():
        # Original flow
        original_embeddings = cell_state_encoder(
            original_indices, original_values, sample_cell_type, original_mask
        )
        original_logits, original_attention = scrna_transformer(
            original_embeddings, original_float_mask
        )

        # Permuted flow
        permuted_embeddings = cell_state_encoder(
            permuted_indices, permuted_values, sample_cell_type, original_mask
        )
        permuted_logits, permuted_attention = scrna_transformer(
            permuted_embeddings, original_float_mask
        )

    # Check attention patterns for each layer
    for layer_idx in range(model_params.num_layers):
        orig_attn = original_attention[layer_idx][0]  # First batch item
        perm_attn = permuted_attention[layer_idx][0]  # First batch item

        # Check CLS token attention if used
        if model_params.use_cls_token:
            check_cls_token_attention(orig_attn, perm_attn, inverse_perm, seq_len, layer_idx)

        # Check sequence-to-sequence attention for each head
        for head_idx in range(model_params.num_heads):
            check_sequence_attention(
                orig_attn,
                perm_attn,
                inverse_perm,
                seq_len,
                layer_idx,
                head_idx,
                model_params.use_cls_token,
            )

    # Finally, check that the prediction remains unchanged
    assert torch.allclose(
        original_logits, permuted_logits, atol=1e-5
    ), "Transformer output logits changed after gene permutation"
