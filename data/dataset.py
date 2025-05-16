import torch
from .scdata import scDATA
from typing import Tuple
from torch.utils.data import Dataset
from sklearn.preprocessing import RobustScaler


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
        self.metadata["cell_type"] = self.metadata.cellsubtype.apply(
            lambda x: self.scdata.cell_types.index(x.lower())
        )
        self.scaler = RobustScaler(with_centering=False).fit(self.scdata.train_adata.X)

    def __len__(self) -> int:
        return self.adata.shape[0]

    def __getitem__(
        self,
        index: int,
        normalize: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the data and label for a given index.

        Args:
            index (int): The index of the data point to retrieve.
            normalize (bool, optional): Whether to normalize the data. Defaults to True.
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The data, cell type, and label.
        """
        # Convert the data to a tensor
        data = self.adata.X[index].toarray().ravel()  # Extract as 1D array
        if normalize:
            # Normalize the data - reshape to 2D for transform, then back to 1D
            data = data.reshape(1, -1)  # Make it 2D for scaler
            data = self.scaler.transform(data).ravel()  # Transform and flatten back to 1D
        data = torch.from_numpy(data).float()
        # Convert the cell type index to a one-hot encoded vector
        cell_type = torch.zeros(len(self.scdata.cell_types), dtype=torch.float32)
        cell_type[self.metadata.cell_type.iloc[index]] = 1.0
        # Convert the label to a tensor
        label = torch.tensor(self.metadata.label.iloc[index]).to(torch.float32)
        return data, cell_type, label
