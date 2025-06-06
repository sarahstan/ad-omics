"""
scDATA: A module for processing single-cell data 
(specifically, ROSMAP from Sun et al., 2023). Works with 
scPLOT class in scplot.py
"""  
import os
import pyreadr
import scanpy as sc
import pandas as pd
import numpy as np
from umap import UMAP


# Fix for NumPy 2.0+ compatibility with older scanpy
if not hasattr(np, "infty"):
    np.infty = np.inf


class scDATA:
    def __init__(self, data_path: str, verbose: bool = False, debug: bool = False):

        # Specify the data path
        self.data_path = data_path

        # Set verbosity
        self.verbose = verbose

        # Set paths for data files
        self.set_paths()

        # Set debug status
        self.debug = debug

        self.adata = None  # Will hold AnnData object
        self.metadata = None
        self.load_data()
        self.load_metadata()
        self.set_cell_types()
        # Initialize attributes to store processed data
        self.adata_hvg = None  # Will store processed data with highly variable genes
        self.embedding_df = None  # Will store dimensional reduction results
        self.gene_stats = None  # Will store gene statistics
        # Define cell type markers as a subset of those used in the paper, specific to celltype
        self.markers = {
            "Endothelial": ["CLDN5", "PECAM1", "ABCB1", "SLC2A1", "CTNNB1"],
            "Ependymal": ["TTR", "HTR2C", "TRMT9B", "CCDC153", "FOXJ1"],
            "Fibroblast": ["COL3A1", "COL1A1", "COL12A1", "MMP2"],
            "Pericyte": ["PDGFRB", "RGS5", "BACH1", "KCNJ8"],
            "Smooth Muscle Cell": ["ACTA2", "MYH11", "TAGLN", "CNN1", "DES", "MYL9"],
        }
        # Filter markers to those present in the dataset
        self.present_markers = [
            gene
            for genes in self.markers.values()
            for gene in genes
            if gene in self.adata.var_names
        ]
        self._print(f"Found {len(self.present_markers)} marker genes in dataset")

    def _print(self, message):
        """
        Print a message to the console.

        Parameters:
        -----------
        message : str
            The message to print
        """
        if self.verbose:
            print(message)

    def set_paths(self):
        """
        Set the paths for the data files.

        Parameters:
        -----------
        data_path : str
            Path to the directory containing the data files
        """
        self.meta_path = os.path.join(self.data_path, "ROSMAP.VascularCells.meta_full.rds")
        self.mtx_path = os.path.join(self.data_path, "counts_matrix.mtx")
        self.genes_path = os.path.join(self.data_path, "gene_names.txt")
        self.barcodes_path = os.path.join(self.data_path, "cell_barcodes.txt")
        # Check that all files exist
        if not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"Metadata file not found: {self.meta_path}")
        if not os.path.exists(self.mtx_path):
            raise FileNotFoundError(f"Matrix Market file not found: {self.mtx_path}")
        if not os.path.exists(self.genes_path):
            raise FileNotFoundError(f"Gene names file not found: {self.genes_path}")
        if not os.path.exists(self.barcodes_path):
            raise FileNotFoundError(f"Cell barcodes file not found: {self.barcodes_path}")

    def load_data(self):
        # Use scanpy to load Matrix Market files
        self.adata = sc.read_mtx(self.mtx_path)
        # Print shape before assigning names
        self._print(f"Matrix shape: {self.adata.shape}")
        # Read gene and cell names
        genes = pd.read_csv(self.genes_path, header=None)
        cells = pd.read_csv(self.barcodes_path, header=None)

        if self.debug: #select only 200 cells, all genes
            self.debug_num = 200
            # Create a copy before slicing to avoid view issues
            self.adata = self.adata.copy()
            self.adata = self.adata[:,:self.debug_num]
            cells = cells[:self.debug_num] 
            print(f'Selected {self.debug_num} cells only for debugging.')

        self._print(f"Number of genes in file: {len(genes)}")
        self._print(f"Number of cells in file: {len(cells)}")
        # check if a transpose needed for alignment
        if self.adata.shape[1] == len(genes) and self.adata.shape[0] == len(cells):
            self.adata.var_names = genes[0].values
            self.adata.obs_names = cells[0].values
            self._print("Successfully assigned gene and cell names!")
        elif self.adata.shape[0] == len(genes) and self.adata.shape[1] == len(cells):
            # Create a copy before transposing
            self.adata = self.adata.copy()
            # Transpose and match
            self.adata = self.adata.T
            self.adata.var_names = genes[0].values
            self.adata.obs_names = cells[0].values
            self._print("Successfully assigned gene and cell names after transpose!")
        else:
            error_str = f"With '{len(genes)}' genes and '{len(cells)}' cells,"
            error_str += f" the data shape is {self.adata.shape}."
            raise ValueError(error_str)
        self._print(f"Final data shape: {self.adata.shape}")
        self._print(f"Sample of var_names: {list(self.adata.var_names[:5])}")
        self._print(f"Sample of obs_names: {list(self.adata.obs_names[:5])}")

    def load_metadata(self):
        self.metadata = pyreadr.read_r(self.meta_path)
        # only one object in the file
        self.metadata = self.metadata[None]
        self._print(f"Original metadata shape: {self.metadata.shape}")
        if self.debug:
            if self.metadata.index.isin(self.adata.obs_names).any():
                self.metadata = self.metadata.loc[self.metadata.index.isin(self.adata.obs_names)]
                print(f'Metadata filtered to match {len(self.metadata)} cells in debug adata.')
        
        self._print("Final Metadata shape:")
        self._print(self.metadata.shape)
        self._print(self.metadata.columns)
        self._print(self.metadata.head())

    def set_cell_types(self):
        self.cell_types = [t.lower() for t in sorted(self.metadata.cellsubtype.unique())]

    def map_cell_type_to_full_name(self, celltype):
        # Maps abbreviated cell types to their full names.
        # Used by add_column_by_column
        mapping = {
            "Endo": "Endothelial",
            "Fib": "Fibroblast",
            "Per": "Pericyte",
            "SMC": "Smooth Muscle Cell",
            "Ependymal": "Ependymal",
        }
        return mapping.get(celltype, celltype)  # Return original if not in mapping

    def add_column_by_column(self, source_col, target_col, mapping_function=None):
        if source_col not in self.metadata.columns:
            raise ValueError(f"Source column '{source_col}' not found in metadata")

        # Determine which mapping function to use based on column names
        if mapping_function is None:
            if source_col == "celltype":
                mapping_function = self.map_cell_type_to_full_name
            else:
                raise ValueError(f"No default mapping function for '{source_col}'")

        # Apply the mapping function to create the new column
        self.metadata[target_col] = self.metadata[source_col].apply(mapping_function)

    def quality_control(self, min_genes=200, min_cells=3, saveQC=False):
        # Store original number of cells and genes
        orig_shape = self.adata.shape
        # Filter cells and genes
        sc.pp.filter_cells(self.adata, min_genes=min_genes)
        sc.pp.filter_genes(self.adata, min_cells=min_cells)
        # Calculate QC metrics
        sc.pp.calculate_qc_metrics(self.adata, inplace=True)
        # Compile summary statistics
        qc_summary = {
            "original_cells": orig_shape[0],
            "original_genes": orig_shape[1],
            "filtered_cells": self.adata.shape[0],
            "filtered_genes": self.adata.shape[1],
            "mean_genes_per_cell": self.adata.obs["n_genes"].mean(),
            "median_genes_per_cell": self.adata.obs["n_genes"].median(),
            "mean_counts_per_cell": self.adata.obs["total_counts"].mean(),
            "median_counts_per_cell": self.adata.obs["total_counts"].median(),
        }

        # Output summary to console
        self._print("QC Summary:")
        self._print(f"Original: {orig_shape[0]} cells x {orig_shape[1]} genes")
        self._print(f"After filtering: {self.adata.shape[0]} cells x {self.adata.shape[1]} genes")
        self._print(f"Mean genes per cell: {qc_summary['mean_genes_per_cell']:.2f}")
        self._print(f"Median genes per cell: {qc_summary['median_genes_per_cell']:.2f}")

        # You can also export QC metrics to a file if needed
        if saveQC:
            self.adata.obs[["n_genes", "total_counts"]].to_csv("qc_metrics_per_cell.csv")

        return qc_summary

    def check_data_preprocessing(self):
        """Analyze if data appears to be already normalized and/or scaled"""
        results = {}

        # Extract a sample of data for efficiency
        if self.adata.n_obs > 1000:
            if hasattr(self.adata.X, "toarray"):  # For sparse matrices
                sample = self.adata.X[:1000].toarray()
            else:
                sample = self.adata.X[:1000]
        else:
            if hasattr(self.adata.X, "toarray"):
                sample = self.adata.X.toarray()
            else:
                sample = self.adata.X

        # Check 1: Look at value distribution
        means = np.mean(sample, axis=0)
        stds = np.std(sample, axis=0)

        # Check for log-transformation
        small_values_ratio = np.mean(sample < 1)
        has_negative = np.any(sample < 0)
        max_value = np.max(sample)
        min_value = np.min(sample)

        results["Distribution"] = {
            "small_values_ratio": small_values_ratio,
            "has_negative_values": has_negative,
            "max_value": max_value,
            "min_value": min_value,
            "mean_of_means": np.mean(means),
            "mean_of_stds": np.mean(stds),
        }

        # Interpretation
        results["Likely_state"] = {
            # Raw counts are integers with no negative values
            "is_raw_counts": not has_negative and np.all(np.mod(sample, 1) == 0) and max_value > 50,
            # Normalized data usually has a target sum but still non-negative
            "is_normalized": not has_negative and small_values_ratio < 0.5 and max_value < 50,
            # Log-transformed data has many small values
            "is_log_transformed": small_values_ratio > 0.5 and max_value < 50,
            # Scaled data typically has mean 0 and std 1, with negative values
            "is_scaled": has_negative and abs(np.mean(means)) < 0.1 and 0.5 < np.mean(stds) < 1.5,
        }

        # Store the analytical results and sample data for plotting
        self.preprocessing_state = results
        self.preprocessing_sample = {"sample": sample, "means": means, "stds": stds}

        return results

    def find_variable_genes(self, n_top_genes=2000, scaleData=True):
        # Use Seurat v3 method for finding variable genes: matching published paper
        sc.pp.highly_variable_genes(
            self.adata,
            flavor="seurat_v3",  # Use Seurat v3 variance stabilization
            n_top_genes=n_top_genes,  # Still select top 2000 genes
            batch_key=None,  # No batch correction
            span=0.3,  # Smoothing parameter similar to Seurat
            subset=False,  # Don't subset the data, just flag variable genes
        )
        # Extract the list of selected genes
        hvg_genes = self.adata.var.index[self.adata.var.highly_variable].tolist()
        self._print(f"Selected {len(hvg_genes)} variable genes using Seurat v3 method")
        # Store statistics in our gene_stats attribute
        self.gene_stats = self.adata.var.copy()
        # Create adata_hvg
        self.adata_hvg = self.adata[:, hvg_genes].copy()
        # Scale data if requested
        if scaleData:
            sc.pp.scale(self.adata_hvg, max_value=10)
        return hvg_genes

    def run_pca(self, use_split=False, split_subset=None, n_comps=30):
        """
        Run PCA on the processed data with highly variable genes.

        Parameters:
        -----------
        n_comps : int, default=60
            Number of principal components to compute
        use_split : Boolean
        split_subset : None or string specifying a computed subset of the total data: train, test, val

        Returns:
        --------
        bool
            True if PCA was successful, False otherwise
        """
        self._print("Running PCA...")

        # Check if we have adata_hvg
        if self.adata_hvg is None:
            print("Error: No processed data found with highly variable genes.")
            print("Please run find_variable_genes() first.")
            return False

        # Check if we have enough cells and genes
        if self.adata_hvg.n_obs < 3 or self.adata_hvg.n_vars < 3:
            error_str = "Error: Not enough data for PCA."
            error_str += f" Found {self.adata_hvg.n_obs} cells and {self.adata_hvg.n_vars} genes."
            print(error_str)
            return False

        # Run PCA
        try:
            if use_split:
                # Run train/test/split to get masks
                train_mask, val_mask, test_mask = self.split_patient_level()
                
                # Apply to adata_hvg and create matching metadata
                if split_subset == "train":
                    pca_input = self.adata_hvg[train_mask].copy()  # Create a copy to avoid view issues
                    metadata_input = self.metadata.loc[train_mask[train_mask].index]
                    subset_name = "training"
                elif split_subset == "test":
                    pca_input = self.adata_hvg[test_mask].copy()
                    metadata_input = self.metadata.loc[test_mask[test_mask].index]
                    subset_name = "test"
                elif split_subset == "val":
                    pca_input = self.adata_hvg[val_mask].copy()
                    metadata_input = self.metadata.loc[val_mask[val_mask].index]
                    subset_name = "validation"
                else:
                    print("Error: split_subset must be train, test, or val.")
                    return False
                    
                self._print(f"Running PCA on {subset_name} subset with {pca_input.n_obs} cells.")
                self._print(f"Corresponding metadata shape: {metadata_input.shape}")
            else: 
                pca_input = self.adata_hvg
                subset_name = "full dataset"

            # Run PCA on the selected input
            sc.tl.pca(pca_input, svd_solver="arpack", n_comps=n_comps)

            # Verify that PCA was successful by checking for X_pca in obsm
            if "X_pca" not in pca_input.obsm:
                print("Error: PCA calculation did not produce expected results.")
                return False
            # Store the result appropriately - either keep it in the subset or transfer to main object
            if use_split:
                # Store the subset as a new attribute
                setattr(self, f"adata_hvg_{split_subset}", pca_input)
                setattr(self, f"metadata_{split_subset}", metadata_input)
                self._print(f"PCA results stored in self.adata_hvg_{split_subset}")
                self._print(f"Metadata stored in self.metadata_{split_subset}")

            self._print(f"PCA completed successfully. Computed {n_comps} principal components.")

            # Print variance explained by first few PCs from the correct object
            variance_ratio = pca_input.uns["pca"]["variance_ratio"]
            cumulative_variance = np.cumsum(variance_ratio)
            self._print(f"Variance explained by first 5 PCs: {variance_ratio[:5].sum():.2%}")
            self._print(f"Variance explained by all PCs: {variance_ratio.sum():.2%}")
            #Fixing a zero indexing error
            pcs_for_90_percent = np.where(cumulative_variance >= 0.9)[0]
            if len(pcs_for_90_percent)>0:
                self._print(f"90% variance explained by first {np.where(cumulative_variance >= 0.9)[0][0] + 1} PCs")

            return True

        except Exception as e:
            print(f"Error running PCA: {e}")
            import traceback

            traceback.print_exc()
            return False

    def run_harmony(self, batch_key, subset = None, max_iter_harmony=20, theta=2, lambda_val=1):
        """
        Run Harmony batch correction on PCA results.

        Parameters:
        -----------
        batch_key : str
            Column in metadata containing batch information
        subset : None or str, default=None
            Specify which dataset to use: None for full dataset, 'train', 'test', or 'val' for specific subsets
        max_iter_harmony : int, default=20
            Maximum number of iterations for Harmony
        theta : float, default=2
            Diversity clustering penalty parameter
        lambda_val : float, default=1
            Ridge regression penalty parameter

        Returns:
        --------
        bool
            True if Harmony correction was successful, False otherwise
        """
        # Determine which dataset to use
        if subset is None:
            # Use full dataset
            input_data = self.adata_hvg
            subset_name = "full dataset"
        else:
            # Use the specified subset if it exists
            subset_attr = f"adata_hvg_{subset}"
            if hasattr(self, subset_attr):
                input_data = getattr(self, subset_attr)
                subset_name = f"{subset} subset"
            else:
                print(f"Error: Subset '{subset}' not found. Available subsets: ")
                # List available subsets
                available_subsets = [attr.replace("adata_hvg_", "") for attr in dir(self) if attr.startswith("adata_hvg_")]
                if available_subsets:
                    print(f"Available subsets: {', '.join(available_subsets)}")
                else:
                    print("No subsets available. Run run_pca with use_split=True first.")
                return False
        
        self._print(f"Running Harmony batch correction on {subset_name} using '{batch_key}'...")

        # Check if the batch key exists
        if batch_key not in self.metadata.columns:
            print(f"Warning: Batch key '{batch_key}' not found in metadata.")
            print(f"Available columns: {list(self.metadata.columns)}")
            return False

        # Make sure we have PCA results in the selected dataset
        if input_data is None or "X_pca" not in input_data.obsm:
            print(f"No PCA results found for {subset_name}.")
            if subset is None:
                print("Running PCA on full dataset first...")
                pca_success = self.run_pca()
                if not pca_success:
                    print("Failed to run PCA. Cannot proceed with Harmony.")
                    return False
                input_data = self.adata_hvg
            else:
                print(f"Please run PCA on the {subset} subset first using run_pca(use_split=True, split_subset='{subset}').")
                return False

        try:
            # import Harmony
            import harmonypy

            # Get PCA matrix from the selected dataset
            pca_matrix = input_data.obsm["X_pca"]
            
            # Get the metadata for the cells in this dataset
            # Filter metadata to only include cells in the input_data
            self._print(f"Input data shape: {input_data.shape}")
            self._print(f"Input data obs index length: {len(input_data.obs.index)}")
            self._print(f"PCA matrix shape: {pca_matrix.shape}")
            self._print(f"Metadata shape: {self.metadata.shape}")
            self._print(f"Sample of input data obs index: {list(input_data.obs.index[:5])}")
            self._print(f"Sample of metadata index: {list(self.metadata.index[:5])}")
            
            # Check if indices overlap
            common_indices = input_data.obs.index.intersection(self.metadata.index)
            self._print(f"Number of overlapping indices: {len(common_indices)}")
            
            if len(common_indices) == 0:
                print("Error: No overlapping cell indices between input data and metadata.")
                print("This suggests a mismatch between cell identifiers.")
                return False
            
            # Filter to common indices only
            try:
                meta_data = self.metadata.loc[common_indices] #, [batch_key]
                self._print(f"Filtered metadata shape: {meta_data.shape}")
                
                # Also filter input data to match
                input_data = input_data[input_data.obs.index.isin(common_indices)].copy()
                self._print(f"Filtered input data shape: {input_data.shape}")                
                
            except Exception as e:
                print(f"Error filtering metadata: {e}")
                print(f"Available columns in metadata: {list(self.metadata.columns)}")
                print(f"Batch key '{batch_key}' in metadata: {batch_key in self.metadata.columns}")
                return False

            # Initialize and run Harmony
            harmony_object = harmonypy.run_harmony(
                data_mat=pca_matrix,
                meta_data=meta_data,
                vars_use=[batch_key],
                theta=theta,  # Diversity clustering penalty
                lamb=lambda_val,  # Ridge regression penalty
                max_iter_harmony=max_iter_harmony,
            )
            
            # double-checking shape
            self._print(f"Z_corr shape: {harmony_object.Z_corr.shape}")
            self._print(f"Expected shape: {input_data.shape[0]} x {input_data.obsm['X_pca'].shape[1]}")
            
            # Store the corrected PCA matrix, transposed
            input_data.obsm["X_pca_harmony"] = harmony_object.Z_corr.T

            self._print(f"Harmony batch correction complete for {subset_name}.")
            
            if subset is None:
                self._print("Harmony embeddings stored in adata_hvg.obsm['X_pca_harmony'].")
            else:
                self._print(f"Harmony embeddings stored in adata_hvg_{subset}.obsm['X_pca_harmony'].")

            # Store references to Harmony datasets for each subset
            if subset is None:
                # Store reference for the full dataset
                self.harmony_input_full = input_data
                # Also set current references for backward compatibility
                self.current_harmony_input = input_data
                self.current_harmony_subset = "full"
            else:
                # Store subset-specific references
                setattr(self, f"harmony_input_{subset}", input_data)
                # Also update the current references
                self.current_harmony_input = input_data
                self.current_harmony_subset = subset

            return True

        except ImportError:
            print("harmonypy package not found. Installing...")
            try:
                import sys
                import subprocess

                subprocess.check_call([sys.executable, "-m", "pip", "install", "harmonypy"])
                import harmonypy

                print("harmonypy installed successfully. Please run this method again.")
            except Exception as e:
                print(f"Failed to install harmonypy: {e}")
                print("Please install harmonypy manually: pip install harmonypy")
            return False

        except Exception as e:
            print(f"Error running Harmony: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_umap(
        self,
        use_harmony=True,
        use_subsets=False,
        n_neighbors=30,
        min_dist=0.3,
        metric="euclidean",
        n_components=2,
        random_state=42,
    ):
        """
        Run UMAP on PCA results or Harmony-corrected PCA.
        If use_subsets=True, fits UMAP on the training subset and transforms test and validation subsets.

        Parameters:
        -----------
        use_harmony : bool, default=True
            Whether to use Harmony-corrected PCA if available
        use_subsets : bool, default=False
            Compute fit_transform for 'train', and just transform for 'test' and 'val' subsets
        n_neighbors : int, default=30
            Number of neighbors to consider for each point
        min_dist : float, default=0.3
            Minimum distance between points in the embedding
        metric : str, default='euclidean'
            Distance metric to use
        n_components : int, default=2
            Number of dimensions for the embedding
        random_state : int, default=42
            Random seed for reproducibility

        Returns:
        --------
        bool
            True if UMAP was successful, False otherwise
        """
        self._print("Running UMAP...")
        
        try:
            # Determine which input data to use based on whether we want subsets
            if use_subsets:
                # Check if we have the required subsets
                required_subsets = ["train", "test", "val"]
                for subset in required_subsets:
                    subset_attr = f"adata_hvg_{subset}"
                    if not hasattr(self, subset_attr):
                        print(f"Error: '{subset}' subset not found. Run PCA with use_split=True first.")
                        print(f"Run: self.run_pca(use_split=True, split_subset='{subset}')")
                        return False
                
                # Determine input data source and PCA embedding key to use
                if use_harmony:
                    # When using Harmony, we get data from harmony_input_{subset}
                    # and look for X_pca_harmony embedding
                    pca_key = "X_pca_harmony"
                    
                    # Check if all harmony inputs exist
                    for subset in required_subsets:
                        harmony_attr = f"harmony_input_{subset}"
                        if not hasattr(self, harmony_attr):
                            print(f"Error: Harmony results not found for {subset} subset.")
                            print(f"Run: self.run_harmony(batch_key='your_batch_key', subset='{subset}')")
                            return False
                        
                        # Check if harmony embeddings exist
                        subset_data = getattr(self, harmony_attr)
                        if pca_key not in subset_data.obsm:
                            print(f"Error: Harmony embeddings not found in {subset} data.")
                            print(f"Something went wrong with Harmony processing for {subset} subset.")
                            return False
                else:
                    # When not using Harmony, we get data from adata_hvg_{subset}
                    # and look for X_pca embedding
                    pca_key = "X_pca"
                    
                    # Check if all subsets exist with PCA results
                    for subset in required_subsets:
                        subset_attr = f"adata_hvg_{subset}"
                        if not hasattr(self, subset_attr):
                            print(f"Error: {subset} subset not found.")
                            print(f"Run: self.run_pca(use_split=True, split_subset='{subset}')")
                            return False
                        
                        # Check if PCA embeddings exist
                        subset_data = getattr(self, subset_attr)
                        if pca_key not in subset_data.obsm:
                            print(f"Error: PCA embeddings not found for {subset} subset.")
                            print(f"Run: self.run_pca(use_split=True, split_subset='{subset}')")
                            return False
                
                self._print(f"Using {'Harmony-corrected' if use_harmony else 'standard'} PCA for UMAP")
                
                # Get train data for fitting
                if use_harmony:
                    train_data = getattr(self, "harmony_input_train")
                else:
                    train_data = getattr(self, "adata_hvg_train")
                    
                X_train = train_data.obsm[pca_key]
                
                # Create UMAP reducer
                umap_reducer = UMAP(
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    metric=metric,
                    n_components=n_components,
                    random_state=random_state,
                )
                
                # Fit UMAP on training data
                self._print("Fitting UMAP on training data...")
                train_embedding = umap_reducer.fit_transform(X_train)
                
                # Store the UMAP model for later use
                self.umap_model = umap_reducer
                
                # Create dataframes for each subset
                train_df = pd.DataFrame(
                    train_embedding,
                    index=train_data.obs.index,
                    columns=[f"UMAP{i+1}" for i in range(n_components)],
                )
                
                # Now transform test and validation data using the fitted model
                subset_embeddings = {}
                subset_dfs = {}
                
                subset_embeddings["train"] = train_embedding
                subset_dfs["train"] = train_df
                
                for subset in ["test", "val"]:
                    # Get the appropriate data for this subset
                    if use_harmony:
                        subset_data = getattr(self, f"harmony_input_{subset}")
                    else:
                        subset_data = getattr(self, f"adata_hvg_{subset}")
                        
                    X_subset = subset_data.obsm[pca_key]
                    
                    self._print(f"Transforming {subset} data using trained UMAP model...")
                    subset_embedding = umap_reducer.transform(X_subset)
                    
                    # Store embeddings
                    subset_embeddings[subset] = subset_embedding
                    
                    # Create dataframe
                    subset_df = pd.DataFrame(
                        subset_embedding,
                        index=subset_data.obs.index,
                        columns=[f"UMAP{i+1}" for i in range(n_components)],
                    )
                    subset_dfs[subset] = subset_df
                
                # Store all embeddings as attributes
                for subset, embedding in subset_embeddings.items():
                    setattr(self, f"umap_embedding_{subset}", embedding)
                
                # Store all dataframes as attributes
                for subset, df in subset_dfs.items():
                    setattr(self, f"embedding_df_{subset}", df)
                
                # Create a combined embedding dataframe with all subsets (optional)
                combined_df = pd.concat([subset_dfs[subset] for subset in required_subsets])
                self.embedding_df = combined_df
                
                # For backward compatibility, store the training embedding as the default one
                self.umap_embedding = train_embedding
                
                self._print("UMAP completed successfully for all subsets.")
                return True
                
            else:
                # Original code for when not using subsets (full dataset)
                # Determine which data source to use
                if use_harmony:
                    if "harmony_input_full" not in dir(self):
                        print("Error: Harmony results for full dataset not found.")
                        print("Run: self.run_harmony(batch_key='your_batch_key')")
                        return False
                    input_data = self.harmony_input_full
                else:
                    input_data = self.adata_hvg

                # Determine which PCA embedding to use
                pca_key = "X_pca_harmony" if use_harmony else "X_pca"
                
                # Check if the required PCA/Harmony results exist
                if pca_key not in input_data.obsm:
                    if use_harmony:
                        print("Error: Harmony-corrected PCA not found for full dataset.")
                        print("Run: self.run_harmony(batch_key='your_batch_key')")
                    else:
                        print("Error: PCA not found for full dataset.")
                        print("Run: self.run_pca()")
                    return False

                if pca_key == "X_pca_harmony":
                    self._print("Using Harmony-corrected PCA for UMAP")
                else:
                    self._print("Using standard PCA for UMAP")

                # Use PCA results as input to reduce noise and computation time
                X_pca = input_data.obsm[pca_key]

                # Create UMAP reducer
                umap_reducer = UMAP(
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    metric=metric,
                    n_components=n_components,
                    random_state=random_state,
                )
    
                # Fit UMAP
                self.umap_embedding = umap_reducer.fit_transform(X_pca)
                
                # Store the UMAP model for later use
                self.umap_model = umap_reducer

                # Create dataframe for visualization for the whole dataset
                self.embedding_df = pd.DataFrame(
                    self.umap_embedding,
                    index=input_data.obs.index,
                    columns=[f"UMAP{i+1}" for i in range(n_components)],
                )
                
                self._print("UMAP completed successfully for the full dataset.")
                return True

        except Exception as e:
            print(f"Error running UMAP: {e}")
            import traceback
            traceback.print_exc()
            return False

    def split_patient_level(
        self,
        test_size=0.15,
        val_size=0.15,
        stratify_cols=["ADdiag2types", "msex"],
        random_state=42,
        copy_data=True,
    ):
        """
        Perform patient-level stratified train/test/validation split.

        Parameters:
        -----------
        test_size : float
            Proportion of patients to include in test set
        val_size : float
            Proportion of patients to include in validation set
        stratify_cols : list
            Columns to use for stratification
        random_state : int
            Random seed for reproducibility
        copy_data : bool
            Whether to create copies of the data (True) or views (False)

        Returns:
        --------
        tuple
            (train_mask, val_mask, test_mask) Boolean masks for each split
        """
        # Replace the global seed with a class property RNG
        self.rng = np.random.RandomState(random_state)

        # Step 1: Create patient-level dataframe with stratification columns
        patient_df = self._create_patient_df(stratify_cols)

        # Step 2: Split patients into train/val/test sets
        train_patients, val_patients, test_patients = self._split_stratified(
            patient_df, stratify_cols, test_size, val_size
        )

        # Step 3: Create masks for cell-level data
        train_mask, val_mask, test_mask = self._create_data_masks(
            train_patients, val_patients, test_patients
        )

        # Step 4: Create split datasets
        self._assign_split_data(train_mask, val_mask, test_mask, copy_data)

        # Step 5: Print summary statistics
        # self._print_split_stats(train_patients, val_patients, test_patients, stratify_cols)

        self.is_split = True
        return train_mask, val_mask, test_mask

    def _create_patient_df(self, stratify_cols):
        """Create a dataframe with one row per patient, aggregating cell-level data."""
        # Create aggregation dictionary
        agg_dict = {}

        # Add columns for stratification
        for col in stratify_cols:
            if col in self.metadata.columns:
                # Use mode for categorical variables
                agg_dict[col] = lambda x: x.mode()[0] if x.mode().shape[0] > 0 else np.nan

        # Group by patient ID and aggregate
        patient_df = self.metadata.groupby("subject").agg(agg_dict).reset_index()

        # Add the count of samples per patient (as a separate step)
        sample_counts = self.metadata.groupby("subject").size()
        patient_df = patient_df.merge(
            sample_counts.rename("sample_count").reset_index(), on="subject", how="left"
        )

        # Add binned sample count for stratification
        patient_df = self._add_sample_count_bins(patient_df)

        # Handle missing values in stratification columns
        patient_df = self._handle_missing_values(patient_df, stratify_cols)

        return patient_df

    def _add_sample_count_bins(self, patient_df):
        """Add binned version of sample counts for stratification."""
        if "sample_count" in patient_df.columns and patient_df["sample_count"].nunique() > 1:
            try:
                # Try to create 4 bins, or fewer if not enough unique values
                n_bins = min(4, patient_df["sample_count"].nunique())
                if n_bins > 1:
                    patient_df["sample_count_bin"] = pd.qcut(
                        patient_df["sample_count"], n_bins, labels=False, duplicates="drop"
                    )
                else:
                    patient_df["sample_count_bin"] = 0
            except Exception as e:
                print(f"Warning: Could not create sample count bins: {e}")
                patient_df["sample_count_bin"] = 0

        return patient_df

    def _handle_missing_values(self, patient_df, stratify_cols):
        """Fill missing values in stratification columns."""
        for col in stratify_cols:
            if col in patient_df.columns and patient_df[col].isna().any():
                # For numeric columns, use median
                if np.issubdtype(patient_df[col].dtype, np.number):
                    fill_value = patient_df[col].median()
                else:
                    # For categorical, use mode
                    fill_value = patient_df[col].mode()[0]

                patient_df[col] = patient_df[col].fillna(fill_value)

        return patient_df

    def _split_stratified(self, patient_df, stratify_cols, test_size, val_size):
        """Split patients into train/val/test sets with stratification."""
        # Define the cutoffs for each split
        train_cutoff = 1.0 - test_size - val_size
        val_cutoff = 1.0 - test_size

        # Initialize lists to hold patients for each split
        train_patients = []
        val_patients = []
        test_patients = []

        # Add sample_count_bin to stratification columns if available
        if "sample_count_bin" in patient_df.columns:
            if "sample_count_bin" not in stratify_cols:
                stratify_cols = stratify_cols + ["sample_count_bin"]

        # Get valid stratification columns (those that exist in the dataframe)
        valid_cols = [col for col in stratify_cols if col in patient_df.columns]

        # Get stratification groups
        if valid_cols:
            # Group by stratification columns
            grouped = patient_df.groupby(valid_cols)

            # For each group, assign patients to splits
            for _, group in grouped:
                patients_in_group = group["subject"].tolist()

                # Split patients in this group
                group_train, group_val, group_test = self._random_assign_group(
                    patients_in_group, train_cutoff, val_cutoff
                )

                # Add to overall lists
                train_patients.extend(group_train)
                val_patients.extend(group_val)
                test_patients.extend(group_test)
        else:
            # If no valid stratification columns, do random split
            print("Warning: No valid stratification columns. Performing random split.")
            train_patients, val_patients, test_patients = self._random_assign_group(
                patient_df["subject"].tolist(), train_cutoff, val_cutoff
            )

        # Check if we missed any patients from patient_df
        all_assigned = set(train_patients) | set(val_patients) | set(test_patients)
        all_patients = set(patient_df["subject"].unique())
        missed_patients = all_patients - all_assigned

        if missed_patients:
            self._print(f"Warning: {len(missed_patients)} patients were not assigned to any split.")
            self._print("Adding them to training set.")
            train_patients.extend(list(missed_patients))

        return train_patients, val_patients, test_patients

    def _random_assign_group(self, patients, train_cutoff, val_cutoff):
        """Randomly assign patients to train/val/test based on cutoffs."""
        # Only modify this line to use the class RNG instead of np.random.random()
        patient_random_values = {patient: self.rng.random() for patient in patients}

        # Assign patients to splits based on random values
        train = [p for p, val in patient_random_values.items() if val < train_cutoff]
        val = [p for p, val in patient_random_values.items() if train_cutoff <= val < val_cutoff]
        test = [p for p, val in patient_random_values.items() if val >= val_cutoff]

        return train, val, test

    def _create_data_masks(self, train_patients, val_patients, test_patients):
        """Create boolean masks for cell-level data based on patient assignments."""
        # Create masks for each split
        train_mask = self.metadata["subject"].isin(train_patients)
        val_mask = self.metadata["subject"].isin(val_patients)
        test_mask = self.metadata["subject"].isin(test_patients)

        # Verify all cells are assigned
        total_cells = self.metadata.shape[0]
        assigned_cells = train_mask.sum() + val_mask.sum() + test_mask.sum()

        if assigned_cells != total_cells:
            # Find subjects that weren't assigned
            all_subjects = set(self.metadata["subject"].unique())
            assigned_subjects = set(train_patients) | set(val_patients) | set(test_patients)
            unassigned_subjects = all_subjects - assigned_subjects

            self._print(
                f"Warning: {total_cells - assigned_cells} cells from {len(unassigned_subjects)} "
                f"subjects were not assigned to any split."
            )

            # Assign unassigned subjects to training set
            self._print("Adding these unassigned subjects to the training set.")
            train_patients = list(train_patients) + list(unassigned_subjects)

            # Update the train mask
            train_mask = self.metadata["subject"].isin(train_patients)

            # Re-verify
            new_assigned = train_mask.sum() + val_mask.sum() + test_mask.sum()
            if new_assigned != total_cells:
                print(f"Error: Still have {total_cells - new_assigned} unassigned cells.")
                # As a last resort, just include all cells
                all_unassigned = ~(train_mask | val_mask | test_mask)
                train_mask = train_mask | all_unassigned

        return train_mask, val_mask, test_mask

    def _assign_split_data(self, train_mask, val_mask, test_mask, copy_data):
        """Assign data to class attributes based on masks."""
        if copy_data:
            # Create copies (safer but uses more memory)
            self.train_adata = self.adata[train_mask].copy()
            self.train_metadata = self.metadata.loc[train_mask].copy()
            self.val_adata = self.adata[val_mask].copy()
            self.val_metadata = self.metadata.loc[val_mask].copy()
            self.test_adata = self.adata[test_mask].copy()
            self.test_metadata = self.metadata.loc[test_mask].copy()
        else:
            # Using views (more memory efficient)
            self.train_adata = self.adata[train_mask]
            self.train_metadata = self.metadata.loc[train_mask]
            self.val_adata = self.adata[val_mask]
            self.val_metadata = self.metadata.loc[val_mask]
            self.test_adata = self.adata[test_mask]
            self.test_metadata = self.metadata.loc[test_mask]

    def _print_split_stats(self, train_patients, val_patients, test_patients, stratify_cols):
        """Print statistics about the train/val/test splits."""
        # Print basic counts
        print(
            f"Training set: {self.train_adata.shape[0]} cells from {len(train_patients)} patients"
        )
        print(f"Validation set: {self.val_adata.shape[0]} cells from {len(val_patients)} patients")
        print(f"Test set: {self.test_adata.shape[0]} cells from {len(test_patients)} patients")

        # Print distribution of stratification variables
        for col in stratify_cols:
            if col in self.metadata.columns:
                print(f"\nDistribution of {col}:")
                print("Train:", self.train_metadata[col].value_counts(normalize=True))
                print("Val:", self.val_metadata[col].value_counts(normalize=True))
                print("Test:", self.test_metadata[col].value_counts(normalize=True))

    def clear_splits(self):
        """Clear the split data to free memory"""
        self.train_adata = None
        self.train_metadata = None
        self.val_adata = None
        self.val_metadata = None
        self.test_adata = None
        self.test_metadata = None

        self.is_split = False
        print("Split data cleared")
    
    def find_discriminative_genes(self,n_genes_de=500,n_genes_final=100,condition_col='ADdiag2types'):
        # Use differential expression to find biologically relevant genes that differ in disease
        # Use Random Forest on DE genes for predictive importance
        # Combine scores and rank
        from sklearn.ensemble import RandomForestClassifier
        import numpy as np

        # First make sure to add condition_col to adata.obs
        self.adata.obs[condition_col] = self.metadata[condition_col]
        # Next, check if data are log-transformed and transform if necessary  
        
        print("Checking data format...")
        print(f"Data type: {self.adata.X.dtype}")
        print(f"Data range before: {self.adata.X.min():.3f} to {self.adata.X.max():.3f}")
        print(f"Is sparse: {hasattr(self.adata.X, 'toarray')}")
        
        # Check if preprocessing is needed
        if self.adata.X.max() > 50:
            print("Preprocessing raw count data...")
            
            # Store original
            self.adata.raw = self.adata.copy()
            
            # Ensure float type
            if hasattr(self.adata.X, 'toarray'):
                self.adata.X = self.adata.X.astype(np.float32)
            else:
                self.adata.X = self.adata.X.astype(np.float32)
            
            # Standard scRNA-seq preprocessing
            sc.pp.normalize_total(self.adata, target_sum=1e4)
            sc.pp.log1p(self.adata)
            
            print(f"Data type after: {self.adata.X.dtype}")
            print(f"Data range after: {self.adata.X.min():.6f} to {self.adata.X.max():.6f}")
            print("✓ Data normalized and log-transformed")
        else:
            print("Data appears to already be processed")

        # 1-Start with differential expression to get biologically relevant genes
        sc.tl.rank_genes_groups(self.adata, condition_col, method='wilcoxon')
        print("OK to ignore prior warning, data is log-transformed")
        de_genes = sc.get.rank_genes_groups_df(self.adata, group=None)   

        #make sure genes are in data
        valid_de_genes = de_genes[de_genes['names'].isin(self.adata.var_names)].copy()
        print(f"DE analysis found {len(de_genes)} genes, {len(valid_de_genes)} are in adata")
        top_de_genes = valid_de_genes.head(min(n_genes_de, len(valid_de_genes)))['names'].tolist()
        print('Found top DE genes.')
        
        #2-Use Random Forest on DE genes for predictive importance
        X_de = self.adata[:, top_de_genes].X
        y_train = self.adata.obs[condition_col]

        #Check shape match
        if X_de.shape[0] != len(y_train):
            raise ValueError(f"Mismatch: X has {X_de.shape[0]} samples, y has {len(y_train)} samples")
    
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_de, y_train)
        print('Random Forest Classifier fitted to data, extracting feature importance.')

        #Get rf_importance from the dataframe
        rf_importance = pd.Series(rf.feature_importances_, index=top_de_genes)
    
        # Get DE scores for the genes we used (from the DataFrame, not the list)
        top_de_genes_df = valid_de_genes.head(min(n_genes_de, len(valid_de_genes)))
        de_scores = top_de_genes_df.set_index('names')['scores']
        
        print('Combining DE and RF scores.')

        # Combine scores (ensure indices align)
        combined_scores = rf_importance * de_scores[rf_importance.index]
        
        final_genes = combined_scores.nlargest(min(n_genes_final, len(combined_scores))).index.tolist()
        self.adata_dg = self.adata[:, final_genes].copy()

        print(f'Top {len(final_genes)} discriminative genes added to adata_dg.')
        
        return final_genes
    
    