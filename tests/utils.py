import torch


def create_permutation(seq_len, seed=None):
    """Create a random permutation of given length."""
    if seed is not None:
        torch.manual_seed(seed)
    return torch.randperm(seq_len)


def create_inverse_permutation(perm):
    """Create the inverse permutation mapping."""
    seq_len = perm.size(0)
    inverse_perm = torch.zeros(seq_len, dtype=torch.long)
    for i, p in enumerate(perm):
        inverse_perm[p] = i
    return inverse_perm


def create_permuted_data(original_data, perm, seq_len, batch_idx=0):
    """Create permuted version of the data for equivariance testing."""
    gene_indices, gene_values, attention_mask = original_data

    # Apply permutation to this sample's data
    sample_indices = gene_indices[batch_idx, :seq_len][perm]
    sample_values = gene_values[batch_idx, :seq_len][perm]

    # Create input tensors with just this single sample
    original_indices = gene_indices[batch_idx : batch_idx + 1].clone()
    original_values = gene_values[batch_idx : batch_idx + 1].clone()
    original_mask = attention_mask[batch_idx : batch_idx + 1].clone()

    # Create permuted tensors
    permuted_indices = original_indices.clone()
    permuted_values = original_values.clone()
    permuted_indices[0, :seq_len] = sample_indices
    permuted_values[0, :seq_len] = sample_values

    return original_indices, original_values, original_mask, permuted_indices, permuted_values


def get_expected_attention_shape(batch_size, num_heads, max_seq_len, use_cls_token):
    """Helper function to determine the expected attention shape."""
    if use_cls_token:
        return (batch_size, num_heads, max_seq_len + 1, max_seq_len + 1)
    else:
        return (batch_size, num_heads, max_seq_len, max_seq_len)


def check_cls_token_attention(orig_attn, perm_attn, inverse_perm, seq_len, layer_idx):
    """Check if CLS token attention is permutation equivariant."""
    # For CLS token attention to sequence tokens
    orig_cls_seq = orig_attn[:, 0, 1 : seq_len + 1]
    perm_cls_seq = perm_attn[:, 0, 1 : seq_len + 1]

    # Apply inverse permutation to permuted attention
    reordered_perm_cls_seq = perm_cls_seq[:, inverse_perm]

    # Check if the attention patterns from CLS token match
    assert torch.allclose(
        orig_cls_seq, reordered_perm_cls_seq, atol=1e-5
    ), f"CLS token attention in layer {layer_idx} is not permutation equivariant"


def check_sequence_attention(
    orig_attn, perm_attn, inverse_perm, seq_len, layer_idx, head_idx, use_cls_token
):
    """Check if sequence-to-sequence attention is permutation equivariant."""
    if use_cls_token:
        # With CLS token, extract sequence-to-sequence attention (excluding CLS token)
        orig_seq_seq = orig_attn[head_idx, 1 : seq_len + 1, 1 : seq_len + 1]
        perm_seq_seq = perm_attn[head_idx, 1 : seq_len + 1, 1 : seq_len + 1]
    else:
        # Without CLS token, check sequence-to-sequence attention directly
        orig_seq_seq = orig_attn[head_idx, :seq_len, :seq_len]
        perm_seq_seq = perm_attn[head_idx, :seq_len, :seq_len]

    # Apply inverse permutation to rows and columns of permuted attention
    reordered_perm_seq_seq = perm_seq_seq[inverse_perm][:, inverse_perm]

    # Check if the attention patterns match
    error_str = f"Sequence-to-sequence attention in layer {layer_idx}, head {head_idx} is not permutation equivariant"
    assert torch.allclose(orig_seq_seq, reordered_perm_seq_seq, atol=1e-5), error_str
