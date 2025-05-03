import os
import scanpy as sc
from torch.utils.data import Dataset

# All datasets that represent a map from keys to data samples should subclass it.
# All subclasses should overwrite __getitem__, supporting fetching a data sample for a given key.
# Subclasses could also optionally overwrite __len__,
# which is expected to return the size of the dataset by many ~torch.utils.data.
# sampler implementations and the default options of ~torch.utils.data.DataLoader.
# Subclasses could also optionally implement __getitems__, for speedup batched samples loading.
# This method accepts list of indices of samples of batch and returns list of samples.


class ADOmicsDataset(Dataset):
    def __init__(self, data_path: str):
        self.data_path = data_path

        # Paths to the files
        self.mtx_file = os.path.join(data_path, "counts_matrix.mtx")
        self.genes_file = os.path.join(data_path, "gene_names.txt")
        self.barcodes_file = os.path.join(data_path, "cell_barcodes.txt")

        # Check if these files exist
        print(f"Matrix file exists: {os.path.exists(self.mtx_file)}")
        print(f"Genes file exists: {os.path.exists(self.genes_file)}")
        print(f"Barcodes file exists: {os.path.exists(self.barcodes_file)}")

        # If they exist, load them
        if (
            os.path.exists(self.mtx_file)
            and os.path.exists(self.genes_file)
            and os.path.exists(self.barcodes_file)
        ):
            # Read the .mtx file
            self.adata = sc.read_mtx(self.mtx_file)

    def __len__(self):
        return self.adata.shape()[0]

    def __getitem__(self, index: int):
        return self.adata.X[index]
