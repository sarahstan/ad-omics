import os
import scanpy as sc
import torch
import pyreadr
from typing import Tuple
from torch.utils.data import Dataset


class ADOmicsDataset(Dataset):
    def __init__(self, data_path: str, subset: str):
        self.data_path = data_path
        self.subset = subset
        self.set_metadata()
        self.set_counts()
        self.set_barcodes()
        self.set_gene_names()
        self.split_subsets()

    def set_counts(self):
        self.mtx_file = os.path.join(self.data_path, "counts_matrix.mtx")
        print(f"Matrix file exists: {os.path.exists(self.mtx_file)}")
        # If they exist, load them
        if os.path.exists(self.mtx_file):
            # Read the .mtx file
            self.counts = sc.read_mtx(self.mtx_file)

    def set_barcodes(self):
        self.barcodes_file = os.path.join(self.data_path, "cell_barcodes.txt")
        print(f"Barcodes file exists: {os.path.exists(self.barcodes_file)}")
        if os.path.exists(self.barcodes_file):
            with open(self.barcodes_file, "r") as file:
                self.barcodes = file.read().split("\n")

    def set_gene_names(self):
        self.genes_file = os.path.join(self.data_path, "gene_names.txt")
        print(f"Genes file exists: {os.path.exists(self.genes_file)}")
        if os.path.exists(self.genes_file):
            with open(self.genes_file, "r") as file:
                self.gene_names = file.read().split("\n")

    def set_metadata(self):
        file_path = f"{self.data_path}/ROSMAP.VascularCells.meta_full.rds"

        result = pyreadr.read_r(file_path)

        # The result is a dictionary where keys are the names of objects
        # and values are pandas DataFrames
        # If there's only one object in the RDS file:
        scRNA_meta = result[None]  # or result[0] depending on the structure

        # Add a new column to spell out the full cell types for visualization
        def longname(celltype):
            if celltype == "Endo":
                return "Endothelial"
            if celltype == "Fib":
                return "Fibroblast"
            if celltype == "Per":
                return "Pericyte"
            if celltype == "SMC":
                return "Smooth Muscle Cell"
            if celltype == "Ependymal":
                return "Ependymal"

        scRNA_meta["celltype"].unique()
        scRNA_meta["celltypefull"] = scRNA_meta["celltype"].apply(longname)
        self.metadata = scRNA_meta

    def split_subsets(self):
        percent_to_use = 0.05
        if self.subset == "train":
            start_percent = 0 * percent_to_use
            end_percent = 1 * percent_to_use
        elif self.subset == "test":
            start_percent = 1 * percent_to_use
            end_percent = 2 * percent_to_use
        elif self.subset == "validation":
            start_percent = 2 * percent_to_use
            end_percent = 3 * percent_to_use
        else:
            raise NotImplementedError("Only train/validation/test subsets are defined.")
        n_samples = self.counts.shape[0]
        start_int = int(start_percent * n_samples)
        end_int = int(end_percent * n_samples)
        self.counts = self.counts[start_int:end_int]
        self.labels = self.metadata["ADdiag2types"][start_int:end_int].apply(
            lambda x: 1.0 if x == "AD" else 0.0
        )

    def __len__(self) -> int:
        return self.counts.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = torch.from_numpy(self.counts.X.toarray()[index])
        label = torch.tensor(self.labels.iloc[index]).to(torch.float32)
        return data, label
