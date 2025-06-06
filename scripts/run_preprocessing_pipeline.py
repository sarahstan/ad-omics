"""
run_preprocessing_pipeline: A module for processing AnnData objects containing gene expression data and cell identity markers.
Works with scDATA and scPLOT types. 
"""     
from data import scDATA
from tools import scPLOT
import numpy as np

def run_pipeline(plot_figs=False):
    #Run complete pipeline on instance of scPLOT generated from scDATA        
    #User: defines directory, creates scDATA instance, uses instance to initialize
    # scPLOT instance, runs this function
    # Specify the data path and figs directory, if saving
    matrix_dir = r'C:\Users\Sarah\Dropbox\Sharejerah\ROSMAP\data'  
    figure_directory= r'C:\Users\Sarah\Dropbox\Bearah Bace\PythonProjects\ad-omics\OtherExports\FIGURES'

    ad_data = scDATA(matrix_dir,verbose=False) # for no print output
    ad_plot = scPLOT(ad_data) #or no save path for no saving

    # Add the full cell type names
    ad_plot.sc_data.add_column_by_column("celltype", "celltypefull")

    # Check pre-processing
    ad_plot.sc_data.check_data_preprocessing()

    # Run quality control
    ad_plot.sc_data.qc_summary = ad_plot.sc_data.quality_control(min_genes=200, min_cells=3)

    if plot_figs:
        # Visualize quality control
        ad_plot.plot_preprocessing_state()

        # Visualize QC metrics
        ad_plot.plot_qc_metrics()
        ad_plot.plot_qc_scatter()

        #Visualize cell distribution for AD dataset
        ad_plot.plot_celltype_composition("celltype")
        ad_plot.plot_meta1_by_meta2("subject", "celltype")
        ad_plot.plot_meta1_by_meta2("ADdiag2types", "celltype")

    #Run dimensionality reduction using PCA
    ad_plot.sc_data.find_variable_genes(n_top_genes=2000)
    ad_plot.sc_data.find_discriminative_genes()
    ad_plot.sc_data.run_pca(n_comps=30)

    #Run Harmony and UMAP using generated PCA
    ad_plot.sc_data.run_harmony(batch_key="batch")
    ad_plot.sc_data.run_umap()

    #Run UMAP on PCA training subset, fit to test/val
    #umap_embedding_trained, umap_embedding_test, umap_embedding_val
    ad_plot.sc_data.run_umap(use_subsets=True)

    if plot_figs:
        #Visualize PCA
        ad_plot.plot_pca()

        #Visualize UMAP in multiple ways
        color_by=["celltype", "cellsubtype", "brain_region", "ADdiag2types", "subject"]
        for meta in color_by:
            ad_plot.plot_umap(color_by=meta)

        #Visualize expression of marker genes using Violin plots
        ad_plot.plot_marker_expression()

        #Visualize marker gene expression by celltype in heatmap
        ad_plot.plot_marker_heatmap()

    print('Run pipeline complete.')