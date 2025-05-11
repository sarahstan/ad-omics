import pytest
from data import scDATA, ADOmicsDataset


@pytest.fixture
def data_path() -> str:
    return r"/mnt/c/Users/JJ/Dropbox/Sharejerah/ROSMAP/data"


@pytest.fixture
def dataset_train(scdata: scDATA) -> ADOmicsDataset:
    dataset = ADOmicsDataset(scdata, subset="train")
    return dataset


@pytest.fixture
def dataset_test(scdata: scDATA) -> ADOmicsDataset:
    dataset = ADOmicsDataset(scdata, subset="test")
    return dataset


@pytest.fixture
def dataset_val(scdata: scDATA) -> ADOmicsDataset:
    dataset = ADOmicsDataset(scdata, subset="val")
    return dataset


def test_dataset(
    dataset_train: ADOmicsDataset,
    dataset_test: ADOmicsDataset,
    dataset_val: ADOmicsDataset,
    test_split: float,
    val_split: float,
) -> None:

    train_split = 1.0 - test_split - val_split

    train_len = len(dataset_train)
    test_len = len(dataset_test)
    val_len = len(dataset_val)

    error_str = "scDATA object does not have attribute train_adata. "
    assert train_len > 0, error_str

    error_str = "scDATA object does not have attribute test_adata. "
    assert test_len > 0, error_str

    error_str = "scDATA object does not have attribute val_adata. "
    assert val_len > 0, error_str

    # Confirm that the dataset lengths are as expected

    total_len = train_len + test_len + val_len

    observed_train_split = train_len / total_len
    observed_test_split = test_len / total_len
    observed_val_split = val_len / total_len

    error_str = (
        f"Train split {observed_train_split:.2f} does not match expected "
        f"{train_split:.2f} within tolerance of 0.05."
    )
    assert abs(observed_train_split - train_split) < 0.05, error_str

    error_str = (
        f"Test split {observed_test_split:.2f} does not match expected "
        f"{test_split:.2f} within tolerance of 0.05."
    )
    assert abs(observed_test_split - test_split) < 0.05, error_str

    error_str = (
        f"Validation split {observed_val_split:.2f} does not match expected "
        f"{val_split:.2f} within tolerance of 0.05."
    )
    assert abs(observed_val_split - val_split) < 0.05, error_str

    breakpoint()
