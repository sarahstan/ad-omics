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
    def __init__(self, data_path: str, verbose: bool = False):

        # Specify the data path
        self.data_path = data_path

        # Set verbosity
        self.verbose = verbose

        # Set paths for data files
        self.set_paths()

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
        self._print(f"Number of genes in file: {len(genes)}")
        self._print(f"Number of cells in file: {len(cells)}")
        # check if a transpose needed for alignment
        if self.adata.shape[1] == len(genes) and self.adata.shape[0] == len(cells):
            self.adata.var_names = genes[0].values
            self.adata.obs_names = cells[0].values
            self._print("Successfully assigned gene and cell names!")
        elif self.adata.shape[0] == len(genes) and self.adata.shape[1] == len(cells):
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
        self._print("Metadata loaded sucessfully.")
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
        print("QC Summary:")
        print(f"Original: {orig_shape[0]} cells x {orig_shape[1]} genes")
        print(f"After filtering: {self.adata.shape[0]} cells x {self.adata.shape[1]} genes")
        print(f"Mean genes per cell: {qc_summary['mean_genes_per_cell']:.2f}")
        print(f"Median genes per cell: {qc_summary['median_genes_per_cell']:.2f}")

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
        print(f"Selected {len(hvg_genes)} variable genes using Seurat v3 method")
        # Store statistics in our gene_stats attribute
        self.gene_stats = self.adata.var.copy()
        # Create adata_hvg
        self.adata_hvg = self.adata[:, hvg_genes].copy()
        # Scale data if requested
        if scaleData:
            sc.pp.scale(self.adata_hvg, max_value=10)
        return hvg_genes

    def run_pca(self, n_comps=30):
        """
        Run PCA on the processed data with highly variable genes.

        Parameters:
        -----------
        n_comps : int, default=60
            Number of principal components to compute

        Returns:
        --------
        bool
            True if PCA was successful, False otherwise
        """
        print("Running PCA...")

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
            sc.tl.pca(self.adata_hvg, svd_solver="arpack", n_comps=n_comps)

            # Verify that PCA was successful by checking for X_pca in obsm
            if "X_pca" not in self.adata_hvg.obsm:
                print("Error: PCA calculation did not produce expected results.")
                return False

            print(f"PCA completed successfully. Computed {n_comps} principal components.")

            # Print variance explained by first few PCs
            variance_ratio = self.adata_hvg.uns["pca"]["variance_ratio"]
            # cumulative_variance = np.cumsum(variance_ratio)
            print(f"Variance explained by first 5 PCs: {variance_ratio[:5].sum():.2%}")
            print(f"Variance explained by all PCs: {variance_ratio.sum():.2%}")

            return True

        except Exception as e:
            print(f"Error running PCA: {e}")
            import traceback

            traceback.print_exc()
            return False

    def run_harmony(self, batch_key, max_iter_harmony=20, theta=2, lambda_val=1):
        """
        Run Harmony batch correction on PCA results.

        Parameters:
        -----------
        batch_key : str
            Column in metadata containing batch information
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
        print(f"Running Harmony batch correction using '{batch_key}'...")

        # Check if the batch key exists
        if batch_key not in self.metadata.columns:
            print(f"Warning: Batch key '{batch_key}' not found in metadata.")
            print(f"Available columns: {list(self.metadata.columns)}")
            return False

        # Make sure we have PCA results
        if self.adata_hvg is None or "X_pca" not in self.adata_hvg.obsm:
            print("No PCA results found. Running PCA first...")
            pca_success = self.run_pca()
            if not pca_success:
                print("Failed to run PCA. Cannot proceed with Harmony.")
                return False

        try:
            # import Harmony
            import harmonypy

            # Get PCA matrix
            pca_matrix = self.adata_hvg.obsm["X_pca"]
            # Get the metadata
            meta_data = self.metadata.loc[self.adata_hvg.obs.index, [batch_key]]

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
            print("Z_corr shape:", harmony_object.Z_corr.shape)
            print(
                "Expected shape:",
                self.adata_hvg.shape[0],
                "x",
                self.adata_hvg.obsm["X_pca"].shape[1],
            )
            # Store the corrected PCA matrix, transposed
            self.adata_hvg.obsm["X_pca_harmony"] = harmony_object.Z_corr.T

            print("Harmony batch correction complete.")
            print("Harmony embeddings stored in adata_hvg.obsm['X_pca_harmony'].")

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
        n_neighbors=30,
        min_dist=0.3,
        metric="euclidean",
        n_components=2,
        random_state=42,
    ):
        """
        Run UMAP on PCA results or Harmony-corrected PCA.

        Parameters:
        -----------
        use_harmony : bool, default=True
            Whether to use Harmony-corrected PCA if available
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
        pandas.DataFrame
            DataFrame with UMAP embeddings
        """
        print("Running UMAP...")

        # Make sure we have PCA results
        if self.adata_hvg is None or "X_pca" not in self.adata_hvg.obsm:
            print("No PCA results found. Running PCA first...")
            self.run_pca()

        try:
            # Determine which PCA embedding to use
            pca_key = (
                "X_pca_harmony"
                if use_harmony and "X_pca_harmony" in self.adata_hvg.obsm
                else "X_pca"
            )

            if pca_key == "X_pca_harmony":
                print("Using Harmony-corrected PCA for UMAP")
            else:
                print("Using standard PCA for UMAP")

            # Use PCA results as input to reduce noise and computation time
            X_pca = self.adata_hvg.obsm[pca_key]

            # Create UMAP reducer
            umap_reducer = UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric=metric,
                n_components=n_components,
                random_state=random_state,
            )

            # Fit UMAP
            umap_embedding = umap_reducer.fit_transform(X_pca)

            # Create dataframe for visualization
            self.embedding_df = pd.DataFrame(
                umap_embedding,
                index=self.adata_hvg.obs.index,
                columns=[f"UMAP{i+1}" for i in range(n_components)],
            )

            return self.embedding_df

        except Exception as e:
            print(f"Error running UMAP: {e}")
            return None

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
