import pytest
from data import scDATA
import torch


@pytest.fixture
def test_split() -> float:
    return 0.15


@pytest.fixture
def val_split() -> float:
    return 0.15


@pytest.fixture
def scdata(test_split: float, val_split: float) -> scDATA:
    """Fixture for creating a SCData instance."""
    # Create a temporary directory for the test
    data_path = "/mnt/c/Users/JJ/Dropbox/Sharejerah/ROSMAP/data"

    # Initialize the SCData instance
    scdata = scDATA(data_path)

    scdata.split_patient_level(
        test_size=test_split,
        val_size=val_split,
        stratify_cols=["ADdiag2types", "msex"],
        random_state=1,
        copy_data=True,
    )

    # Return the instance for use in tests
    return scdata


@pytest.fixture
def generate_gene_data():
    """
    Fixture factory that returns a function to generate random gene data.

    This is a "factory fixture" that returns a function, which can then be called
    with specific parameters from other fixtures.
    """

    def _generate_data(batch_size, num_genes, max_seq_len, seed=42):
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

    return _generate_data
