import torch
from .scdata import scDATA
from typing import Tuple
from torch.utils.data import Dataset


class ADOmicsDataset(Dataset):
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.scdata = scDATA(data_path)

    def __len__(self) -> int:
        return self.counts.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = torch.from_numpy(self.counts.X.toarray()[index])
        label = torch.tensor(self.labels.iloc[index]).to(torch.float32)
        return data, label
