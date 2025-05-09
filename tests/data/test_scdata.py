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


@pytest.fixture
def split_seed_2(scdata: scDATA, test_split: float, val_split: float) -> scDATA:
    """Fixture for creating a SCData instance with a different split seed."""
    # Perform the split with a different random state
    scdata.clear_splits()
    scdata.split_patient_level(
        test_size=test_split,
        val_size=val_split,
        stratify_cols=["ADdiag2types", "msex"],
        random_state=2,
        copy_data=True,
    )

    # Return the instance for use in tests
    return scdata


def test_split_ratios(scdata: scDATA, test_split: float, val_split: float) -> None:
    """Test the train-validation-test split ratios of the dataset."""

    # Calculate the expected number of samples in each set
    num_samples = len(scdata.metadata)
    train_size = scdata.train_metadata.shape[0]
    train_fraction = train_size / num_samples
    val_size = scdata.val_metadata.shape[0]
    val_fraction = val_size / num_samples
    test_size = scdata.test_metadata.shape[0]
    test_fraction = test_size / num_samples

    expected_train_fraction = 1 - test_split - val_split
    expected_val_fraction = val_split
    expected_test_fraction = test_split

    # Check that each fraction is within a 5% tolerance of the expected value
    tolerance = 0.03

    error_str = (
        f"Train fraction {train_fraction:.2f} does not match expected "
        f"{expected_train_fraction:.2f} within tolerance of {tolerance:.2f}."
    )
    assert abs(train_fraction - expected_train_fraction) < tolerance, error_str

    error_str = (
        f"Validation fraction {val_fraction:.2f} does not match expected "
        f"{expected_val_fraction:.2f} within tolerance of {tolerance:.2f}."
    )
    assert abs(val_fraction - expected_val_fraction) < tolerance, error_str

    error_str = (
        f"Test fraction {test_fraction:.2f} does not match expected "
        f"{expected_test_fraction:.2f} within tolerance of {tolerance:.2f}."
    )
    assert abs(test_fraction - expected_test_fraction) < tolerance, error_str

    error_str = "All examples sum to number of samples."
    assert train_size + val_size + test_size == num_samples, error_str

    breakpoint()


def test_split_disjoint(scdata: scDATA) -> None:
    """Test the train-validation-test split of the dataset."""

    # Ensure there are no patients in both train and test sets
    train_patients = set(scdata.train_metadata["subject"])
    test_patients = set(scdata.test_metadata["subject"])
    val_patients = set(scdata.val_metadata["subject"])

    error_str = "Train and test sets should not share patients."
    assert train_patients.isdisjoint(test_patients), error_str

    error_str = "Train and validation sets should not share patients."
    assert train_patients.isdisjoint(val_patients), error_str

    error_str = "Validation and test sets should not share patients."
    assert val_patients.isdisjoint(test_patients), error_str
