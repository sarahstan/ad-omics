import pytest
import torch
from data import scDATA, VectorData, TokenData
from data.dataset import ADOmicsDataset
from torch.utils.data import DataLoader


@pytest.fixture
def batch_size() -> int:
    return 32


@pytest.fixture
def dataset_vector(scdata: scDATA) -> ADOmicsDataset:
    return ADOmicsDataset(scDATA=scdata, subset="train", representation="vector")


@pytest.fixture
def dataset_token(scdata: scDATA) -> ADOmicsDataset:
    return ADOmicsDataset(scDATA=scdata, subset="train", representation="tokens")


@pytest.fixture
def dataloader_shuffled_vector(dataset_vector: ADOmicsDataset, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset_vector,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(1),
        collate_fn=VectorData.collate,
    )


@pytest.fixture
def dataloader_unshuffled_vector(dataset_vector: ADOmicsDataset, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset_vector,
        batch_size=batch_size,
        shuffle=False,
        generator=torch.Generator().manual_seed(2),
        collate_fn=VectorData.collate,
    )


@pytest.fixture
def dataloader_shuffled_token(dataset_token: ADOmicsDataset, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset_token,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(1),
        collate_fn=TokenData.collate,
    )


@pytest.fixture
def dataloader_unshuffled_token(dataset_token: ADOmicsDataset, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset_token,
        batch_size=batch_size,
        shuffle=False,
        generator=torch.Generator().manual_seed(2),
        collate_fn=TokenData.collate,
    )


def test_dataloader_shuffling_vector(
    dataloader_shuffled_vector: DataLoader,
    dataloader_unshuffled_vector: DataLoader,
) -> None:
    # Extract two batches from each dataloader

    # Shuffled  dataloader
    batch_shuffled = []
    for e, b in enumerate(dataloader_shuffled_vector):
        if e < 2:
            batch_shuffled.append(b)
        else:
            break
    genes_shuffled = torch.concat([b[0] for b in batch_shuffled])
    cell_types_shuffled = torch.concat([b[1] for b in batch_shuffled])
    labels_shuffled = torch.concat([b[2] for b in batch_shuffled])

    # Unshuffled dataloader
    batch_unshuffled = []
    for e, b in enumerate(dataloader_unshuffled_vector):
        if e < 2:
            batch_unshuffled.append(b)
        else:
            break

    genes_unshuffled = torch.concat([b[0] for b in batch_unshuffled])
    cell_types_unshuffled = torch.concat([b[1] for b in batch_unshuffled])
    labels_unshuffled = torch.concat([b[2] for b in batch_unshuffled])

    error_str = "genes_shuffled and genes_unshuffled should not be identical."
    assert not torch.equal(genes_shuffled, genes_unshuffled), error_str

    error_str = "cell_types_shuffled and cell_types_unshuffled should not be identical."
    assert not torch.equal(cell_types_shuffled, cell_types_unshuffled), error_str

    error_str = "labels_shuffled and labels_unshuffled should not be identical."
    assert not torch.equal(labels_shuffled, labels_unshuffled), error_str


def test_dataloader_shuffling_token(
    dataloader_shuffled_token: DataLoader,
    dataloader_unshuffled_token: DataLoader,
) -> None:
    # Extract two batches from each dataloader

    # Shuffled  dataloader
    batch_shuffled = []
    for e, b in enumerate(dataloader_shuffled_token):
        if e < 2:
            batch_shuffled.append(b)
        else:
            break
    breakpoint()
    gene_ids_shuffled = torch.concat([b[0] for b in batch_shuffled])
    gene_counts_shuffled = torch.concat([b[1] for b in batch_shuffled])
    cell_types_shuffled = torch.concat([b[2] for b in batch_shuffled])
    labels_shuffled = torch.concat([b[3] for b in batch_shuffled])

    # Unshuffled dataloader
    batch_unshuffled = []
    for e, b in enumerate(dataloader_unshuffled_token):
        if e < 2:
            batch_unshuffled.append(b)
        else:
            break

    gene_ids_unshuffled = torch.concat([b[0] for b in batch_unshuffled])
    gene_counts_unshuffled = torch.concat([b[1] for b in batch_unshuffled])
    cell_types_unshuffled = torch.concat([b[2] for b in batch_unshuffled])
    labels_unshuffled = torch.concat([b[3] for b in batch_unshuffled])

    error_str = "gene_ids_shuffled and gene_ids_unshuffled should not be identical."
    assert not torch.equal(gene_ids_shuffled, gene_ids_unshuffled), error_str

    error_str = "gene_counts_shuffled and gene_counts_unshuffled should not be identical."
    assert not torch.equal(gene_counts_shuffled, gene_counts_unshuffled), error_str

    error_str = "cell_types_shuffled and cell_types_unshuffled should not be identical."
    assert not torch.equal(cell_types_shuffled, cell_types_unshuffled), error_str

    error_str = "labels_shuffled and labels_unshuffled should not be identical."
    assert not torch.equal(labels_shuffled, labels_unshuffled), error_str
