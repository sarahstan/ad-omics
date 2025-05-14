import pytest
import torch
import numpy as np
from data import scDATA, ADOmicsDataset
from torch.utils.data import DataLoader
from scipy.stats import ks_2samp


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
    dataset: ADOmicsDataset,
    batch_size: int,
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

    breakpoint()
