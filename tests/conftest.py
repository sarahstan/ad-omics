# Updated conftest.py
import pytest
from data import scDATA
import torch
from dataclasses import dataclass
from models.torch.cell_state_encoder import CellStateEncoder


@dataclass
class ModelParameters:
    """Common parameters for all model tests."""

    def __init__(
        self,
        # Shared
        batch_size: int = 8,
        embed_dim: int = 16,  # Embedding dimension
        # Cell State Encoder parameters
        num_genes_total: int = 15000,  # Total vocabulary size
        num_cell_types: int = 12,  # Number of cell types
        num_genes_per_cell_max: int = 500,  # Maximum number of genes
        use_film: bool = True,  # Use FiLM for cell type conditioning
        cell_state_encoder_dropout: float = 0.1,  # Dropout rate for cell state encoder
        # Transformer parameters
        num_heads: int = 8,  # Number of attention heads
        ff_dim: int = 64,  # Feed forward dimension
        num_layers: int = 2,  # Number of transformer layers
        transformer_dropout: float = 0.1,  # Dropout rate for transformer
    ) -> None:
        """Initialize the common parameters for model tests."""
        # Set attributes from the init parameters
        for key, value in locals().items():
            if key != "self":
                setattr(self, key, value)


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
def model_params(scdata) -> ModelParameters:
    """Common fixture for model parameters used across all model tests."""
    params = ModelParameters()
    params.num_cell_types = len(scdata.cell_types)
    return params


@pytest.fixture
def gene_token_data(model_params):
    """Fixture to create token representation of gene data for all tests."""
    from tests.utils import generate_gene_data

    return generate_gene_data(
        batch_size=model_params.batch_size,
        num_genes_total=model_params.num_genes_total,
        num_genes_per_cell_max=model_params.num_genes_per_cell_max,
    )


@pytest.fixture
def cell_type(model_params):
    """Fixture to create a tensor of cell types for all tests."""
    # Create an integer tensor representing cell types
    torch.manual_seed(42)  # For reproducibility
    cell_type = torch.randint(0, model_params.num_cell_types, (model_params.batch_size,))
    return cell_type


@pytest.fixture
def cell_state_encoder(model_params) -> CellStateEncoder:
    """Fixture to create a CellStateEncoder instance for testing."""
    return CellStateEncoder(
        num_genes_total=model_params.num_genes_total,
        gene_embedding_dim=model_params.embed_dim,
        num_cell_types=model_params.num_cell_types,
        use_film=model_params.use_film,
        dropout=model_params.cell_state_encoder_dropout,
    )
