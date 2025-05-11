import torch
from .scdata import scDATA
from typing import Tuple
from torch.utils.data import Dataset


class ADOmicsDataset(Dataset):
    def __init__(self, scDATA: scDATA, subset: str = "train") -> None:
        """_summary_

        Args:
            data_path (str): The path to the data directory.
            subset (str, optional): The subset of the data to load. Defaults to "train".

        Raises:
            ValueError: If the subset is not recognized.
        """
        self.scdata = scDATA
        self.subset = subset
        self.load_subset()

    def load_subset(self) -> None:
        # Ensure scdata has already done the split
        attrs = [
            "train_adata",
            "test_adata",
            "val_adata",
            "train_metadata",
            "test_metadata",
            "val_metadata",
        ]
        for attr in attrs:
            if not hasattr(self.scdata, attr):
                error_str = f"scDATA object does not have attribute {attr}. "
                error_str += "Split with scdata.split_patient_level()"
                raise ValueError(error_str)
        self.adata = getattr(self.scdata, f"{self.subset}_adata")
        self.metadata = getattr(self.scdata, f"{self.subset}_metadata")
        self.metadata["label"] = (self.metadata.ADdiag2types == "AD").astype(int)

    def __len__(self) -> int:
        return self.adata.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = torch.from_numpy(self.adata.X.toarray()[index])
        label = torch.tensor(self.metadata.label.iloc[index]).to(torch.float32)
        return data, label
