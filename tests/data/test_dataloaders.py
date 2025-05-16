import pytest
import torch
from data import scDATA, ADOmicsDataset
from torch.utils.data import DataLoader


@pytest.fixture
def batch_size() -> int:
    return 32


@pytest.fixture
def dataset(scdata: scDATA) -> ADOmicsDataset:
    return ADOmicsDataset(scDATA=scdata, subset="train")


@pytest.fixture
def dataloader_shuffled(dataset: ADOmicsDataset, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(1),
    )


@pytest.fixture
def dataloader_unshuffled(dataset: ADOmicsDataset, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        generator=torch.Generator().manual_seed(2),
    )


def test_dataloader_shuffling(
    dataloader_shuffled: DataLoader,
    dataloader_unshuffled: DataLoader,
) -> None:
    # Extract two batches from each dataloader

    # Shuffled  dataloader
    batch_shuffled = []
    for e, b in enumerate(dataloader_shuffled):
        if e < 2:
            batch_shuffled.append(b)
        else:
            break
    genes_shuffled = torch.concat([b[0] for b in batch_shuffled])
    cell_types_shuffled = torch.concat([b[1] for b in batch_shuffled])
    labels_shuffled = torch.concat([b[2] for b in batch_shuffled])

    # Unshuffled dataloader
    batch_unshuffled = []
    for e, b in enumerate(dataloader_unshuffled):
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
