import pytest
from data import scDATA


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
