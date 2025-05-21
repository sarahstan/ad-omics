import torch
from typing import Tuple, Union
from .scdata import scDATA
from .types import VectorData, TokenData
from torch.utils.data import Dataset
from sklearn.preprocessing import RobustScaler
import enum


class Representation(enum.Enum):
    VECTOR = "vector"
    TOKENS = "tokens"


class ADOmicsDataset(Dataset):
    def __init__(
        self,
        scDATA: scDATA,
        subset: str = "train",
        representation: str = "vector",
    ) -> None:
        """
        Initialize the ADOmicsDataset.

        Args:
            data_path (str): The path to the data directory.
            subset (str, optional): The subset of the data to load. Defaults to "train".
            representation (str, optional): The gene representation to use. Defaults to "vector".

        Raises:
            ValueError: If the subset is not recognized.
        """
        self.scdata = scDATA
        self.subset = subset
        if representation == "vector":
            self.representation = Representation.VECTOR
        elif representation == "tokens":
            self.representation = Representation.TOKENS
        else:
            error_str = f"Invalid representation: {representation}. " "Choose 'vector' or 'tokens'."
            raise ValueError(error_str)
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

    def _get_gene_vector(self, index: int, normalize: bool = True) -> torch.Tensor:
        data = self.adata.X[index].toarray().ravel()  # Extract as 1D array
        if normalize:
            # Normalize the data - reshape to 2D for transform, then back to 1D
            data = data.reshape(1, -1)  # Make it 2D for scaler
            data = self.scaler.transform(data).ravel()  # Transform and flatten back to 1D
        data = torch.from_numpy(data).float()
        return data

    def _get_gene_tokens(
        self, index: int, normalize: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.adata.X[index].toarray().ravel()
        if normalize:
            # Normalize the data - reshape to 2D for transform, then back to 1D
            data = data.reshape(1, -1)
            data = self.scaler.transform(data).ravel()
        data = torch.from_numpy(data).float()
        # Return a 2D-tensor of shape [non-zero index, count]
        non_zero_indices = data.nonzero().ravel()
        non_zero_values = data[non_zero_indices]
        return non_zero_indices, non_zero_values

    def _get_cell_type_one_hot(self, index: int) -> torch.Tensor:
        # Convert the cell type index to a one-hot encoded vector
        cell_type = torch.zeros(len(self.scdata.cell_types), dtype=torch.float32)
        cell_type[self.metadata.cell_type.iloc[index]] = 1.0
        return cell_type

    def _get_cell_type_token(self, index: int) -> torch.Tensor:
        # Convert the cell type index to a one-hot encoded vector
        cell_type = self.metadata.cell_type.iloc[index]
        return torch.tensor(cell_type, dtype=torch.long)

    def __getitem__(
        self,
        index: int,
        normalize: bool = True,
    ) -> Union[VectorData, TokenData]:
        """
        Get the data and label for a given index.

        Args:
            index (int): The index of the data point to retrieve.
            normalize (bool, optional): Whether to normalize the data. Defaults to True.

        Returns:
            Union[VectorData, TokenData]: The data, cell type, and label.
        """
        # Convert the label to a tensor
        label = torch.tensor(self.metadata.label.iloc[index]).to(torch.float32)
        # Convert the data to a tensor
        if self.representation == Representation.VECTOR:
            gene_vector = self._get_gene_vector(index, normalize)
            cell_type = self._get_cell_type_one_hot(index)
            return VectorData(
                gene_vector=gene_vector,
                cell_type=cell_type,
                label=label,
            )
        elif self.representation == Representation.TOKENS:
            gene_tokens, gene_counts = self._get_gene_tokens(index, normalize)
            cell_type = self._get_cell_type_token(index)
            return TokenData(
                gene_indices=gene_tokens,
                gene_counts=gene_counts,
                cell_type=cell_type,
                label=label,
            )
        else:
            error_str = f"Invalid gene representation: {self.representation}. "
            error_str += "Choose 'vector' or 'tokens'."
            raise ValueError(error_str)
