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


def get_expected_attention_shape(batch_size, num_heads, max_seq_len):
    """Helper function to determine the expected attention shape."""
    return (batch_size, num_heads, max_seq_len, max_seq_len)


def check_sequence_attention(orig_attn, perm_attn, inverse_perm, seq_len, layer_idx, head_idx):
    """Check if sequence-to-sequence attention is permutation equivariant."""
    # Check sequence-to-sequence attention directly
    orig_seq_seq = orig_attn[head_idx, :seq_len, :seq_len]
    perm_seq_seq = perm_attn[head_idx, :seq_len, :seq_len]

    # Apply inverse permutation to rows and columns of permuted attention
    reordered_perm_seq_seq = perm_seq_seq[inverse_perm][:, inverse_perm]

    # Check if the attention patterns match
    error_str = f"Sequence-to-sequence attention in layer {layer_idx}, "
    error_str += f"head {head_idx} is not permutation equivariant. "
    assert torch.allclose(orig_seq_seq, reordered_perm_seq_seq, atol=1e-5), error_str


def generate_gene_data(batch_size, num_genes, max_seq_len, seed=42):
    """Generate random gene expression data for testing."""
    torch.manual_seed(seed)  # For reproducibility

    # Generate unique random gene indices for each sample in the batch
    gene_indices = []
    gene_values = []

    for _ in range(batch_size):
        # Sample a random number of expressed genes (between 100 and max_seq_len)
        num_expressed = torch.randint(100, max_seq_len, (1,)).item()

        # Sample random gene indices (without replacement)
        indices = torch.randperm(num_genes)[:num_expressed]
        gene_indices.append(indices)

        # Generate random expression values for each gene
        values = torch.abs(torch.randn(num_expressed))
        gene_values.append(values)

    # Create attention mask and pad to max_seq_len
    padded_indices = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    padded_values = torch.zeros(batch_size, max_seq_len, dtype=torch.float)
    attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)

    for i, (indices, values) in enumerate(zip(gene_indices, gene_values)):
        seq_len = indices.size(0)
        padded_indices[i, :seq_len] = indices
        padded_values[i, :seq_len] = values
        attention_mask[i, :seq_len] = True

    return padded_indices, padded_values, attention_mask
