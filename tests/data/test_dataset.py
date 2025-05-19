import pytest
import torch
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


@pytest.fixture
def dataset_train_tokens(scdata: scDATA) -> ADOmicsDataset:
    dataset = ADOmicsDataset(scdata, subset="train", representation="tokens")
    return dataset


def test_dataset_lengths(
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


def test_dataset_get(
    dataset_train: ADOmicsDataset,
    dataset_test: ADOmicsDataset,
    dataset_val: ADOmicsDataset,
) -> None:
    # Test the __getitem__ method
    train_example = dataset_train[0]
    gene_vector = train_example.gene_vector
    cell_type = train_example.cell_type
    label = train_example.label
    data_shape = gene_vector.shape
    cell_type_shape = cell_type.shape
    label_shape = label.shape

    test_example = dataset_test[0]
    assert test_example.gene_vector.shape == data_shape, "Data shape is incorrect"
    assert test_example.cell_type.shape == cell_type_shape, "Cell type shape is incorrect"
    assert test_example.label.shape == label_shape, "Label shape is incorrect"

    val_example = dataset_val[0]
    assert val_example.gene_vector.shape == data_shape, "Data shape is incorrect"
    assert val_example.cell_type.shape == cell_type_shape, "Cell type shape is incorrect"
    assert val_example.label.shape == label_shape, "Label shape is incorrect"


def test_cell_type_representation(
    dataset_train: ADOmicsDataset,
    dataset_test: ADOmicsDataset,
    dataset_val: ADOmicsDataset,
) -> None:
    # Test the cell type representation

    for dataset in [dataset_train, dataset_test, dataset_val]:
        example = dataset[0]
        cell_type_vec = example.cell_type
        cell_type_int = torch.argmax(cell_type_vec).item()

        error_str = "There should be only one non-zero cell type in the example."
        assert cell_type_vec.sum() == 1, error_str

        error_str = "The cell type integer representation is incorrect."
        assert cell_type_int in range(len(dataset.scdata.cell_types)), error_str


def test_token_representation(
    dataset_train_tokens: ADOmicsDataset,
) -> None:
    # Randomly select 10 examples from the dataset
    for i in range(10):
        example = dataset_train_tokens[i]

        error_str = "The gene indices and values should have the same length."
        assert len(example.gene_indices) == len(example.gene_counts), error_str

        error_str = "The cell type should be an integer in [0, number of cell type])."
        assert example.cell_type in range(len(dataset_train_tokens.scdata.cell_types)), error_str
