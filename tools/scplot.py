# scPLOT

"""
scPLOT: A module for visualizing single-cell data 
(specifically, ROSMAP from Sun et al., 2023) with Plotly.
Works with scDATA class in scdata.py
"""  

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scanpy as sc
import numpy as np
from scipy import sparse
from scipy.stats import zscore
import pandas as pd


class scPLOT:
    """
    A class for visualizing single-cell data with Plotly.
    
    Parameters
    ----------
    sc_data : AnnData
        Scanpy AnnData object containing single-cell data.
    
    Run the full pipeline like this:
    --------
    >>> ad_path = 'your/data/path'
    >>> ad_data = scDATA(ad_path,verbose=True) # for print output
    >>> fig_directory= 'your/fig/path'
    >>> ad_plot = scPLOT(ad_data,fig_directory) # to save figs
    >>> ad_plot.run_pipeline(plot_figs=True)
    """
    def __init__(self, sc_data, save_path=""):
        self.sc_data = sc_data
        self.save_path=save_path
        #if a save path, save=true

    def save_fig(self, fig, path_with_extension):
        if self.save_path:
            fig.write_image(path_with_extension)

    def plot_qc_metrics(self):
        # save path should end filename.png
        # Plot QC violin plots
        # Create subplots
        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=(
                "RNA counts per cell",
                "Genes detected per cell",
                "Mitochondrial content (%)",
            ),
        )
        # RNA counts
        fig.add_trace(
            go.Violin(
                y=self.sc_data.metadata["nCount_RNA"],
                box_visible=True,
                line_color="black",
                fillcolor="lightseagreen",
                opacity=0.6,
                name="RNA counts",
            ),
            row=1,
            col=1,
        )
        # Number of genes
        fig.add_trace(
            go.Violin(
                y=self.sc_data.metadata["nFeature_RNA"],
                box_visible=True,
                line_color="black",
                fillcolor="darkorange",
                opacity=0.6,
                name="Gene count",
            ),
            row=1,
            col=2,
        )
        # Mitochondrial percentage
        fig.add_trace(
            go.Violin(
                y=self.sc_data.metadata["percent.mt"],
                box_visible=True,
                line_color="black",
                fillcolor="indianred",
                opacity=0.6,
                name="MT %",
            ),
            row=1,
            col=3,
        )
        # Update layout
        fig.update_layout(height=400, width=1000, title_text="QC Metrics Distribution")
        # Save fig as PNG
        if self.save_path:
            path_with_extension = self.save_path + "QC_Metrics.png"
            self.save_fig(fig,path_with_extension)
        return fig

    def plot_qc_scatter(self):
        # save path should end filename.png``
        # Create a figure with 2 subplots
        fig = make_subplots(
            rows=1, cols=2, subplot_titles=("RNA counts vs Genes", "Mitochondrial % vs Genes")
        )
        # RNA counts vs number of genes
        fig.add_trace(
            go.Scatter(
                x=self.sc_data.metadata["nFeature_RNA"],
                y=self.sc_data.metadata["nCount_RNA"],
                mode="markers",
                marker=dict(color="lightseagreen", opacity=0.5, size=5),
                name="Cells",
            ),
            row=1,
            col=1,
        )
        # Mitochondrial percentage vs number of genes
        fig.add_trace(
            go.Scatter(
                x=self.sc_data.metadata["nFeature_RNA"],
                y=self.sc_data.metadata["percent.mt"],
                mode="markers",
                marker=dict(color="indianred", opacity=0.5, size=5),
                name="Cells",
            ),
            row=1,
            col=2,
        )
        # Update layout
        fig.update_layout(height=400, width=900, title_text="QC Relationships")
        # Update axes
        fig.update_xaxes(title_text="Genes detected", row=1, col=1)
        fig.update_yaxes(title_text="RNA counts", row=1, col=1)
        fig.update_xaxes(title_text="Genes detected", row=1, col=2)
        fig.update_yaxes(title_text="Mitochondrial %", row=1, col=2)
        # Save fig as PNG
        if self.save_path:
            path_with_extension = self.save_path + "QC_Scatter.png"
            self.save_fig(fig,path_with_extension)
        return fig

    def plot_celltype_composition(self, cellcolumn):
        # Count cells by celltype (cellcolumn should be celltypefull or celltype)
        celltype_counts = self.sc_data.metadata[cellcolumn].value_counts().reset_index()
        celltype_counts.columns = [cellcolumn, "count"]

        # Create pie chart
        fig = px.pie(
            celltype_counts,
            values="count",
            names=cellcolumn,
            title="Cell Type Composition",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(height=500, width=700)
        # Save fig as PNG
        if self.save_path:
            path_with_extension = self.save_path + "Celltype_Composition_" + cellcolumn + ".png"
            self.save_fig(fig,path_with_extension)
        return fig

    def plot_meta1_by_meta2(self, col1, col2, title=None):
        # Group by col1 and col2, barplot
        # Ex: 'subject', 'celltype'
        grouped = self.sc_data.metadata.groupby([col1, col2]).size().reset_index(name="count")
        # Use custom title if provided, otherwise create a default one
        if title is None:
            title = f"{col2} Distribution by {col1}"
        # Create stacked bar chart
        fig = px.bar(
            grouped,
            x=col1,
            y="count",
            color=col2,
            title=title,
            color_discrete_sequence=px.colors.qualitative.Bold,
        )

        fig.update_layout(height=500, width=900)
        # Save fig as PNG
        if self.save_path:
            path_with_extension = self.save_path + "Plot_" + col1 + "_by_" + col2 + ".png"
            self.save_fig(fig,path_with_extension)
        return fig

    def plot_preprocessing_state(self):
        # Visualize the preprocessing state of the data
        # Check if analysis has been run
        if not hasattr(self.sc_data, "preprocessing_state"):
            print("Please run sc_data.check_data_preprocessing() first")
            return
        # Get the data from the scDATA object
        results = self.sc_data.preprocessing_state
        sample = self.sc_data.preprocessing_sample["sample"]
        means = self.sc_data.preprocessing_sample["means"]
        stds = self.sc_data.preprocessing_sample["stds"]
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Distribution of Values",
                "Distribution of Gene Means",
                "Distribution of Gene Standard Deviations",
                "Expression Distribution for Sample Genes",
            ),
        )
        # Plot 1: Histogram of all values
        fig.add_trace(go.Histogram(x=sample.flatten(), nbinsx=50, name="All Values"), row=1, col=1)
        # Plot 2: Distribution of means
        fig.add_trace(go.Histogram(x=means, nbinsx=30, name="Gene Means"), row=1, col=2)
        # Plot 3: Distribution of standard deviations
        fig.add_trace(go.Histogram(x=stds, nbinsx=30, name="Gene SDs"), row=2, col=1)
        # Plot 4: Expression values for a few random genes
        gene_indices = np.random.choice(sample.shape[1], min(5, sample.shape[1]), replace=False)
        for idx in gene_indices:
            # Create KDE-like distribution
            gene_data = sample[:, idx]
            # Use a faster approximation of KDE with histogram
            fig.add_trace(
                go.Histogram(
                    x=gene_data, nbinsx=30, histnorm="probability density", name=f"Gene {idx}"
                ),
                row=2,
                col=2,
            )
        # Update layout
        fig.update_layout(
            height=800, width=1000, title_text="Preprocessing State Analysis", showlegend=True
        )
        # Update axes labels
        fig.update_xaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_xaxes(title_text="Mean Expression", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_xaxes(title_text="Standard Deviation", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_xaxes(title_text="Expression Value", row=2, col=2)
        fig.update_yaxes(title_text="Density", row=2, col=2)
        # Display the figure
        fig.show()
        # Print preprocessing state summary
        print("\nData appears to be:")
        for state, is_true in results["Likely_state"].items():
            state_name = state.replace("is_", "").replace("_", " ").title()
            if is_true:
                print(f"- {state_name}: Yes")
            else:
                print(f"- {state_name}: No")

        # Save fig as PNG
        if self.save_path:
            path_with_extension = self.save_path + "Plot_Preprocessing_State.png"
            self.save_fig(fig,path_with_extension)
        return fig

    def plot_pca(self, color_by="celltype"):
        # Plot PCA colored by metadata
        pca_df = pd.DataFrame(
            self.sc_data.adata_hvg.obsm["X_pca_harmony"][:, :2],
            index=self.sc_data.adata_hvg.obs.index,
            columns=["PC1", "PC2"],
        )
        pca_df[color_by] = self.sc_data.metadata.loc[pca_df.index, color_by]

        fig = px.scatter(
            pca_df,
            x="PC1",
            y="PC2",
            color=color_by,
            title="PCA Visualization",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig.update_traces(marker=dict(size=5, opacity=0.7))
        fig.update_layout(height=600, width=800)
        fig.show()
        # Save fig as PNG
        if self.save_path:
            path_with_extension = self.save_path + "Plot_PCA_by_" + color_by + ".png"
            self.save_fig(fig,path_with_extension)
        return fig

    def plot_umap(self, color_by="celltype"):
        # Plot UMAP (self.sc_data.embedding_df) colored by metadata
        # some color_by options = 'celltype','brain_region','ADdiag2types','subject'

        # Add necessary metadata
        self.sc_data.embedding_df[color_by] = self.sc_data.metadata.loc[
            self.sc_data.embedding_df.index, color_by
        ]

        # create dynamic title
        title = f"UMAP Visualization: by {color_by}"
        # Create plot
        fig = px.scatter(
            self.sc_data.embedding_df,
            x="UMAP1",
            y="UMAP2",
            color=color_by,
            title=title,
            color_discrete_sequence=px.colors.qualitative.Bold,
        )

        fig.update_traces(marker=dict(size=5, opacity=0.7))
        fig.update_layout(height=600, width=800)
        fig.show()

        # Save fig as PNG
        if self.save_path:
            path_with_extension = self.save_path + "Plot_UMAP_by_" + color_by + ".png"
            self.save_fig(fig,path_with_extension)
        return fig

    def plot_marker_expression(self, cell_type_col="celltype"):
        # Generate violin plots of expression for celltype marker genes defined in scDATA class
        if not self.sc_data.present_markers:
            print("No marker genes found in the dataset.")
            return
        # Create expression dataframe
        expr_df = pd.DataFrame(index=self.sc_data.adata.obs.index)
        for gene in self.sc_data.present_markers:
            if sparse.issparse(self.sc_data.adata.X):
                expr_df[gene] = self.sc_data.adata[:, gene].X.toarray().flatten()
            else:
                expr_df[gene] = self.sc_data.adata[:, gene].X.flatten()
        # Add cell type information
        expr_df[cell_type_col] = self.sc_data.metadata.loc[expr_df.index, cell_type_col]
        # Create violin plots for markers by cell type
        for cell_type, genes in self.sc_data.markers.items():
            genes_in_data = [g for g in genes if g in self.sc_data.adata.var_names]
            if not genes_in_data:
                continue
            fig = make_subplots(
                rows=1,
                cols=len(genes_in_data),
                subplot_titles=[f"{gene}" for gene in genes_in_data],
            )
            for i, gene in enumerate(genes_in_data):
                for j, ct in enumerate(expr_df[cell_type_col].unique()):
                    y_vals = expr_df.loc[expr_df[cell_type_col] == ct, gene]
                    fig.add_trace(
                        go.Violin(
                            y=y_vals,
                            name=ct,
                            showlegend=i == 0,  # Only show legend on first plot
                            box_visible=True,
                            meanline_visible=True,
                            line_color=px.colors.qualitative.Bold[
                                j % len(px.colors.qualitative.Bold)
                            ],
                        ),
                        row=1,
                        col=i + 1,
                    )
            fig.update_layout(
                title=f"{cell_type} Marker Expression by Cell Type",
                height=500,
                width=max(300 * len(genes_in_data), 600),
            )
            fig.show()
        # Save fig as PNG
        if self.save_path:
            path_with_extension = self.save_path + "Plot_Marker_Violin_by_" + cell_type_col + ".png"
            self.save_fig(fig,path_with_extension)
        return fig

    def plot_marker_heatmap(self, cell_type_col="celltype"):
        # Create heatmap of marker gene expression by celltype
        if not self.sc_data.present_markers:
            print("No marker genes found in the dataset.")
            return
        # Create expression dataframe
        expr_df = pd.DataFrame(index=self.sc_data.adata.obs.index)
        for gene in self.sc_data.present_markers:
            if sparse.issparse(self.sc_data.adata.X):
                expr_df[gene] = self.sc_data.adata[:, gene].X.toarray().flatten()
            else:
                expr_df[gene] = self.sc_data.adata[:, gene].X.flatten()
        # Add cell type information
        expr_df[cell_type_col] = self.sc_data.metadata.loc[expr_df.index, cell_type_col]
        # Calculate mean expression by cell type
        self.marker_means = expr_df.groupby(cell_type_col)[self.sc_data.present_markers].mean()
        # Z-score
        self.marker_zscores = self.marker_means.apply(zscore, axis=0, nan_policy="omit")
        # Create heatmap
        fig = px.imshow(
            self.marker_zscores,
            labels={"x": "Gene", "y": "Cell Type", "color": "Z-score"},
            color_continuous_scale="RdBu_r",
            title="Cell Type Marker Expression",
        )
        fig.update_layout(height=600, width=900)
        fig.show()
        # Save fig as PNG
        if self.save_path:
            path_with_extension = self.save_path + "Plot_Marker_Heatmap_by_" + cell_type_col + ".png"
            self.save_fig(fig,path_with_extension)
        return fig
        
    def run_pipeline(self,plot_figs=False):
        #Run complete pipeline on instance of scPLOT generated from scDATA        
        #User: defines directory, creates scDATA instance, uses instance to initialize
        # scPLOT instance, runs this function

        # Add the full cell type names
        self.sc_data.add_column_by_column("celltype", "celltypefull")

        # Check pre-processing
        self.sc_data.check_data_preprocessing()

        # Run quality control
        self.sc_data.qc_summary = self.sc_data.quality_control(min_genes=200, min_cells=3)

        if plot_figs:
            # Visualize quality control
            self.plot_preprocessing_state()

            # Visualize QC metrics
            self.plot_qc_metrics()
            self.plot_qc_scatter()

            #Visualize cell distribution for AD dataset
            self.plot_celltype_composition("celltype")
            self.plot_meta1_by_meta2("subject", "celltype")
            self.plot_meta1_by_meta2("ADdiag2types", "celltype")

        #Run dimensionality reduction using PCA
        self.sc_data.find_variable_genes(n_top_genes=2000)
        self.sc_data.run_pca(n_comps=30)

        #Run Harmony and UMAP using generated PCA
        self.sc_data.run_harmony(batch_key="batch")
        self.sc_data.run_umap()

        if plot_figs:
            #Visualize PCA
            self.plot_pca()

            #Visualize UMAP in multiple ways
            color_by=["celltype", "cellsubtype", "brain_region", "ADdiag2types", "subject"]
            for meta in color_by:
                self.plot_umap(color_by=meta)

            #Visualize expression of marker genes using Violin plots
            self.plot_marker_expression()

            #Visualize marker gene expression by celltype in heatmap
            self.plot_marker_heatmap()

        print('Run pipeline complete.')