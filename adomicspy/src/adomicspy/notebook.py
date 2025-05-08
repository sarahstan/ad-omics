# src/ad_omics/loader.py

import scanpy as sc
import pandas as pd
import os
from adomicspy.scdata import scDATA
from adomicspy.scplot import scPLOT

def load_data(
    matrix_dir=None,
    meta_file=None,
    mtx_file=None,
    genes_file=None,
    barcodes_file=None
):
    """
    Load ROSMAP scRNA-seq data.
    
    Parameters
    ----------
    matrix_dir : str, optional
        Directory containing the data files. If provided, other file paths
        will be constructed relative to this unless explicitly specified.
    meta_file : str, optional
        Path to the metadata RDS file. If None but matrix_dir is provided,
        defaults to "ROSMAP.VascularCells.meta_full.rds" in matrix_dir.
    mtx_file : str, optional
        Path to the count matrix file. If None but matrix_dir is provided,
        defaults to "counts_matrix.mtx" in matrix_dir.
    genes_file : str, optional
        Path to the gene names file. If None but matrix_dir is provided,
        defaults to "gene_names.txt" in matrix_dir.
    barcodes_file : str, optional
        Path to the cell barcodes file. If None but matrix_dir is provided,
        defaults to "cell_barcodes.txt" in matrix_dir.
    
    Returns
    -------
    ad_data : scDATA
        A scDATA object with the loaded data.
    """
    # Set default paths if matrix_dir is provided
    if matrix_dir is not None:
        if meta_file is None:
            meta_file = os.path.join(matrix_dir, "ROSMAP.VascularCells.meta_full.rds")
        if mtx_file is None:
            mtx_file = os.path.join(matrix_dir, "counts_matrix.mtx")
        if genes_file is None:
            genes_file = os.path.join(matrix_dir, "gene_names.txt")
        if barcodes_file is None:
            barcodes_file = os.path.join(matrix_dir, "cell_barcodes.txt")
    
    # Initialize the single-cell data object
    ad_data = scDATA(mtx_file, meta_file, genes_file, barcodes_file)
    
    # Add the full cell type names
    ad_data.add_column_by_column('celltype', 'celltypefull')
    
    return ad_data
def preprocess_data(adata, min_genes=200, min_cells=3):
    """
    Preprocess and perform quality control on scRNA-seq data.
    
    Parameters
    ----------
    adata : scDATA
        The scDATA object to preprocess
    min_genes : int, default=200
        Minimum number of genes required for a cell to pass QC
    min_cells : int, default=3
        Minimum number of cells required for a gene to pass QC
    
    Returns
    -------
    qc_summary : dict
        Summary of the quality control metrics
    """
    # Run preprocessing
    adata.check_data_preprocessing()
    
    # Run quality control
    qc_summary = adata.quality_control(min_genes=min_genes, min_cells=min_cells)
    
    return qc_summary

def load_and_preprocess(
    matrix_dir=None,
    meta_file=None,
    mtx_file=None,
    genes_file=None,
    barcodes_file=None,
    min_genes=200,
    min_cells=3
):
    """
    Load and preprocess ROSMAP scRNA-seq data in one step.
    
    Parameters
    ----------
    [all parameters from both load_data and preprocess_data]
    
    Returns
    -------
    adata : scDATA
        Processed scDATA object
    qc_summary : dict
        Summary of quality control metrics
    """
    # Load data
    adata = load_data(
        matrix_dir=matrix_dir,
        meta_file=meta_file,
        mtx_file=mtx_file,
        genes_file=genes_file,
        barcodes_file=barcodes_file
    )
    
    # Preprocess data
    qc_summary = preprocess_data(adata, min_genes=min_genes, min_cells=min_cells)
    
    return adata, qc_summary

def visualize_qc(adata, show_plots=True):
    """
    Visualize quality control metrics for the scRNA-seq data.
    
    Parameters
    ----------
    adata : scDATA
        The processed scDATA object
    show_plots : bool, default=True
        Whether to immediately display the plots
    
    Returns
    -------
    plots : dict
        Dictionary of generated plot objects
    """
    # Initialize the plotting object
    ad_plot = scPLOT(adata)
    
    # Store plots in a dictionary
    plots = {}
    
    # Visualize preprocessing results
    plots['preprocessing'] = ad_plot.plot_preprocessing_state()
    
    # Visualize QC metrics
    plots['qc_metrics'] = ad_plot.plot_qc_metrics()
    plots['qc_scatter'] = ad_plot.plot_qc_scatter()
    
    # Show plots if requested
    if show_plots:
        for plot in plots.values():
            if hasattr(plot, 'show'):
                plot.show()
    
    return plots


def visualize_cell_distribution(adata, show_plots=True):
    """
    Visualize cell type distribution and composition.
    
    Available metadata fields include:
    ['subject', 'orig.ident', 'nCount_RNA', 'nFeature_RNA', 'id',
    'library_id', 'batch', 'brain_region', 'age_death', 'msex', 'pmi',
    'ADdiag2types', 'percent.mt', 'percent.rp', 'celltype', 'cellsubtype']
    
    Parameters
    ----------
    adata : scDATA
        The processed scDATA object
    show_plots : bool, default=True
        Whether to immediately display the plots
    
    Returns
    -------
    plots : dict
        Dictionary of generated plot objects
    """
    # Initialize the plotting object
    ad_plot = scPLOT(adata)
    
    # Store plots in a dictionary
    plots = {}
    
    # Cell type distribution
    plots['celltype_composition'] = ad_plot.plot_celltype_composition('celltype')
    plots['subject_by_celltype'] = ad_plot.plot_meta1_by_meta2('subject', 'celltype')
    plots['diagnosis_by_celltype'] = ad_plot.plot_meta1_by_meta2('ADdiag2types', 'celltype')
    
    # Show plots if requested
    if show_plots:
        for plot in plots.values():
            if hasattr(plot, 'show'):
                plot.show()
    
    return plots


def visualize_custom_metadata(adata, metadata1, metadata2, show_plots=True):
    """
    Visualize custom metadata relationships.
    
    Parameters
    ----------
    adata : scDATA
        The processed scDATA object
    metadata1 : str
        First metadata field to visualize
    metadata2 : str
        Second metadata field to visualize
    show_plots : bool, default=True
        Whether to immediately display the plots
    
    Returns
    -------
    plot : object
        The generated plot object
    """
    # Initialize the plotting object
    ad_plot = scPLOT(adata)
    
    # Create the custom metadata plot
    plot = ad_plot.plot_meta1_by_meta2(metadata1, metadata2)
    
    # Show plot if requested
    if show_plots and hasattr(plot, 'show'):
        plot.show()
    
    return plot

def run_dimensionality_reduction(adata, n_top_genes=2000, n_comps=30):
    """
    Perform variable gene selection and PCA dimensionality reduction.
    
    Parameters
    ----------
    adata : scDATA
        The processed scDATA object
    n_top_genes : int, default=2000
        Number of highly variable genes to select
    n_comps : int, default=30
        Number of principal components to compute
    
    Returns
    -------
    adata : scDATA
        The scDATA object with PCA results added
    """
    # Find variable genes
    adata.find_variable_genes(n_top_genes=n_top_genes)
    
    # Run PCA
    adata.run_pca(n_comps=n_comps)
    
    return adata

def run_harmony_and_umap(adata, batch_key='batch'):
    """
    Perform batch correction using Harmony and generate UMAP embedding.
    
    Parameters
    ----------
    adata : scDATA
        The scDATA object with PCA results
    batch_key : str, default='batch'
        Column in adata.obs containing batch information
    
    Returns
    -------
    adata : scDATA
        The scDATA object with Harmony and UMAP results added
    """
    # Run Harmony batch correction
    adata.run_harmony(batch_key=batch_key)
    
    # Generate UMAP embedding
    adata.run_umap()
    
    return adata

def run_complete_reduction_pipeline(adata, n_top_genes=2000, n_comps=30, batch_key='batch'):
    """
    Run the complete dimensionality reduction pipeline:
    1. Find variable genes
    2. Run PCA
    3. Perform batch correction with Harmony
    4. Generate UMAP embedding
    
    Parameters
    ----------
    adata : scDATA
        The processed scDATA object
    n_top_genes : int, default=2000
        Number of highly variable genes to select
    n_comps : int, default=30
        Number of principal components to compute
    batch_key : str, default='batch'
        Column in adata.obs containing batch information
    
    Returns
    -------
    adata : scDATA
        The scDATA object with all dimensionality reduction results
    """
    # Variable genes and PCA
    adata = run_dimensionality_reduction(adata, n_top_genes=n_top_genes, n_comps=n_comps)
    
    # Harmony and UMAP
    adata = run_harmony_and_umap(adata, batch_key=batch_key)
    
    return adata

def visualize_pca(adata, show_plots=True):
    """
    Visualize PCA components.
    
    Parameters
    ----------
    adata : scDATA
        The scDATA object with PCA results
    show_plots : bool, default=True
        Whether to immediately display the plots
    
    Returns
    -------
    plot : object
        The PCA plot object
    """
    # Initialize the plotting object
    ad_plot = scPLOT(adata)
    
    # Generate PCA plot
    plot = ad_plot.plot_pca()
    
    # Show plot if requested
    if show_plots and hasattr(plot, 'show'):
        plot.show()
    
    return plot


def visualize_umap(adata, color_by=['celltype', 'cellsubtype', 'brain_region', 'ADdiag2types', 'subject'], 
                   show_plots=True):
    """
    Visualize UMAP embeddings colored by different metadata.
    
    Parameters
    ----------
    adata : scDATA
        The scDATA object with UMAP results
    color_by : str or list, default=['celltype', 'cellsubtype', 'brain_region', 'ADdiag2types', 'subject']
        Metadata column(s) to use for coloring the UMAP
    show_plots : bool, default=True
        Whether to immediately display the plots
    
    Returns
    -------
    plots : dict
        Dictionary of UMAP plots keyed by the metadata used for coloring
    """
    # Initialize the plotting object
    ad_plot = scPLOT(adata)
    
    # Convert single string to list if necessary
    if isinstance(color_by, str):
        color_by = [color_by]
    
    # Generate UMAP plots colored by different metadata
    plots = {}
    for meta in color_by:
        plots[meta] = ad_plot.plot_umap(color_by=meta)
        
        # Show plot if requested
        if show_plots and hasattr(plots[meta], 'show'):
            plots[meta].show()
    
    return plots


def visualize_dimensional_reductions(adata, umap_colors=['celltype'], show_plots=True):
    """
    Visualize both PCA and UMAP results.
    
    Parameters
    ----------
    adata : scDATA
        The scDATA object with PCA and UMAP results
    umap_colors : list, default=['celltype']
        List of metadata columns to use for coloring UMAPs
    show_plots : bool, default=True
        Whether to immediately display the plots
    
    Returns
    -------
    plots : dict
        Dictionary containing all generated plots
    """
    plots = {'pca': visualize_pca(adata, show_plots=show_plots)}
    plots['umap'] = visualize_umap(adata, color_by=umap_colors, show_plots=show_plots)
    
    return plots

def visualize_marker_expression(adata, show_plots=True):
    """
    Visualize expression of marker genes using violin plots.
    
    Parameters
    ----------
    adata : scDATA
        The processed scDATA object
    show_plots : bool, default=True
        Whether to immediately display the plots
    
    Returns
    -------
    plot : object
        The marker expression plot object
    """
    # Initialize the plotting object
    ad_plot = scPLOT(adata)
    
    # Generate marker expression plot
    plot = ad_plot.plot_marker_expression()
    
    # Show plot if requested
    if show_plots and hasattr(plot, 'show'):
        plot.show()
    
    return plot


def visualize_marker_heatmap(adata, show_plots=True):
    """
    Visualize marker gene expression using a heatmap.
    
    Parameters
    ----------
    adata : scDATA
        The processed scDATA object
    show_plots : bool, default=True
        Whether to immediately display the plots
    
    Returns
    -------
    plot : object
        The marker heatmap plot object
    """
    # Initialize the plotting object
    ad_plot = scPLOT(adata)
    
    # Generate marker heatmap
    plot = ad_plot.plot_marker_heatmap()
    
    # Show plot if requested
    if show_plots and hasattr(plot, 'show'):
        plot.show()
    
    return plot

def run_complete_visualization(adata, show_plots=True):
    """
    Run the complete visualization pipeline:
    1. QC metrics
    2. Cell type distributions
    3. PCA and UMAP embeddings
    4. Marker gene expression
    
    Parameters
    ----------
    adata : scDATA
        The fully processed scDATA object
    show_plots : bool, default=True
        Whether to immediately display the plots
    
    Returns
    -------
    plots : dict
        Dictionary containing all generated plots
    """
    plots = {}
    
    # QC visualizations
    plots['qc'] = visualize_qc(adata, show_plots=show_plots)
    
    # Cell type distributions
    plots['cell_distribution'] = visualize_cell_distribution(adata, show_plots=show_plots)
    
    # Dimensional reductions
    plots['dimensional_reduction'] = visualize_dimensional_reductions(
        adata,
        umap_colors=['celltype', 'cellsubtype', 'brain_region', 'ADdiag2types', 'subject'],
        show_plots=show_plots
    )
    
    # Marker gene expression
    plots['markers'] = visualize_all_markers(adata, show_plots=show_plots)
    
    return plots


# If this module is run directly (not imported), load data with default paths
if __name__ == "__main__":
    
    # Load and preprocess data
    default_matrix_dir = r'C:\Users\Sarah\Dropbox\Sharejerah\ROSMAP\data'
    ad_data, _ = load_and_preprocess(matrix_dir=default_matrix_dir)
    
    # Visualize QC
    qc_plots = visualize_qc(ad_data)
    
    # Visualize cell distributions
    cell_plots = visualize_cell_distribution(ad_data)
    
    # Example of custom metadata visualization
    custom_plot = visualize_custom_metadata(ad_data, 'batch', 'celltype')

    # Run complete dimensionality reduction pipeline
    ad_data = run_complete_reduction_pipeline(ad_data)
    print("Analysis pipeline complete with PCA, Harmony, and UMAP")
    
    # Run all visualizations
    all_plots = run_complete_visualization(ad_data)
    
    print("Entire pipeline completed successfully")