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
            self.print("Please run sc_data.check_data_preprocessing() first")
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
        self.sc_data._print("\nData appears to be:")
        for state, is_true in results["Likely_state"].items():
            state_name = state.replace("is_", "").replace("_", " ").title()
            if is_true:
                self.sc_data._print(f"- {state_name}: Yes")
            else:
                self.sc_data._print(f"- {state_name}: No")

        # Save fig as PNG
        if self.save_path:
            path_with_extension = self.save_path + "Plot_Preprocessing_State.png"
            self.save_fig(fig,path_with_extension)
        return fig

    def plot_pca(self, subset=None, color_by="celltype", use_harmony=True, components=[0, 1]):
        """
        Plot PCA visualization colored by metadata.
        
        Parameters:
        -----------
        subset : None or str, default=None
            Specify which subset to plot: None for full dataset, 'train', 'test', or 'val'
        color_by : str, default="celltype"
            Column from metadata to use for coloring points
        use_harmony : bool, default=True
            Whether to use Harmony-corrected PCA or standard PCA
        components : list, default=[0, 1]
            Which principal components to plot (0-indexed)
            
        Returns:
        --------
        plotly.graph_objects.Figure
            The generated plot
        """
        # Determine which dataset to use
        if subset is None:
            # Use full dataset
            if use_harmony:
                if not hasattr(self.sc_data, "harmony_input_full"):
                    print("Error: Harmony results for full dataset not found.")
                    print("Run: self.sc_data.run_harmony(batch_key='your_batch_key')")
                    return None
                input_data = self.sc_data.harmony_input_full
                subset_name = "full dataset"
            else:
                input_data = self.sc_data.adata_hvg
                subset_name = "full dataset"
        else:
            # Use the specified subset if it exists
            if use_harmony:
                harmony_attr = f"harmony_input_{subset}"
                if not hasattr(self.sc_data, harmony_attr):
                    print(f"Error: Harmony results for {subset} subset not found.")
                    print(f"Run: self.sc_data.run_harmony(batch_key='your_batch_key', subset='{subset}')")
                    return None
                input_data = getattr(self.sc_data, harmony_attr)
                subset_name = f"{subset} subset"
            else:
                adata_attr = f"adata_hvg_{subset}"
                if not hasattr(self.sc_data, adata_attr):
                    print(f"Error: {subset} subset not found.")
                    print(f"Run: self.sc_data.run_pca(use_split=True, split_subset='{subset}')")
                    return None
                input_data = getattr(self.sc_data, adata_attr)
                subset_name = f"{subset} subset"
        
        # Determine which PCA embedding to use
        pca_key = "X_pca_harmony" if use_harmony else "X_pca"
        
        # Check if the embedding exists
        if pca_key not in input_data.obsm:
            correction_type = "Harmony" if use_harmony else "PCA"
            print(f"Error: {correction_type} results not found for {subset_name}.")
            return None
        
        # Extract component labels
        pc1_idx, pc2_idx = components
        pc1_label = f"PC{pc1_idx+1}"
        pc2_label = f"PC{pc2_idx+1}"
        
        # Create dataframe for plotting
        pca_df = pd.DataFrame(
            input_data.obsm[pca_key][:, components],
            index=input_data.obs.index,
            columns=[pc1_label, pc2_label],
        )
        
        # Add metadata for coloring
        if color_by in self.sc_data.metadata.columns:
            try:
                pca_df[color_by] = self.sc_data.metadata.loc[pca_df.index, color_by]
            except KeyError:
                # Handle case where some indices might not be in metadata
                print(f"Warning: Some cells in the {subset_name} are not found in metadata.")
                # Get intersection of indices
                common_indices = pca_df.index.intersection(self.sc_data.metadata.index)
                pca_df = pca_df.loc[common_indices]
                pca_df[color_by] = self.sc_data.metadata.loc[common_indices, color_by]
        else:
            print(f"Warning: Column '{color_by}' not found in metadata.")
            if 'celltype' in self.sc_data.metadata.columns:
                color_by = 'celltype'
                pca_df[color_by] = self.sc_data.metadata.loc[pca_df.index, color_by]
            else:
                # Create a dummy coloring variable
                color_by = 'group'
                pca_df[color_by] = 'All Cells'

        # Create title based on parameters
        correction_type = "Harmony-corrected" if use_harmony else "standard"
        title = f"PCA Visualization ({correction_type}) - {subset_name.capitalize()}"
        
        # Create plot
        fig = px.scatter(
            pca_df,
            x=pc1_label,
            y=pc2_label,
            color=color_by,
            title=title,
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        
        # Add variance explained if available
        if hasattr(input_data, 'uns') and 'pca' in input_data.uns and 'variance_ratio' in input_data.uns['pca']:
            variance_ratio = input_data.uns['pca']['variance_ratio']
            pc1_var = variance_ratio[pc1_idx] * 100
            pc2_var = variance_ratio[pc2_idx] * 100
            fig.update_layout(
                xaxis_title=f"{pc1_label} ({pc1_var:.1f}% variance)",
                yaxis_title=f"{pc2_label} ({pc2_var:.1f}% variance)"
            )
        
        # Update styling
        fig.update_traces(marker=dict(size=5, opacity=0.7))
        fig.update_layout(
            height=600, 
            width=800,
            template="plotly_white",
            legend_title_text=color_by.capitalize()
        )
        
        # Show the figure
        fig.show()
        
        # Save figure if save path is specified
        if self.save_path:
            subset_str = f"_{subset}" if subset else ""
            harmony_str = "_harmony" if use_harmony else ""
            path_with_extension = f"{self.save_path}Plot_PCA{subset_str}{harmony_str}_by_{color_by}.png"
            self.save_fig(fig, path_with_extension)
        
        return fig

    def plot_umap(self, use_split=False, color_by="celltype"):
        # Plot UMAP (self.sc_data.embedding_df) colored by metadata
        # Use Split enables plotting UMAP generated from training data and overlaying Test/Val data
        # some color_by options = 'celltype','brain_region','ADdiag2types','subject','traintestsplit'

        #Determine if split and set variables
        if use_split:
            #Create embedding_df that concatenates train/test/split and adds a column for train/test/split identity
            # Convert each UMAP output to a dataframe with proper column names
            df1 = pd.DataFrame(self.sc_data.embedding_df_train, columns=["UMAP1", "UMAP2"])
            df2 = pd.DataFrame(self.sc_data.embedding_df_test, columns=["UMAP1", "UMAP2"])
            df3 = pd.DataFrame(self.sc_data.embedding_df_val, columns=["UMAP1", "UMAP2"])

            # Add the traintestsplit column to each dataframe
            df1["traintestsplit"] = "train"
            df2["traintestsplit"] = "test"
            df3["traintestsplit"] = "val"

            # Concatenate all three dataframes
            embedding_df = pd.concat([df1, df2, df3], ignore_index=True)

            # Verify the result
            print(f"Combined shape: {embedding_df.shape}")
            print(embedding_df.head())
            print(embedding_df.tail())

        else:
            embedding_df = self.sc_data.embedding_df
            # Add necessary metadata
            embedding_df[color_by] = self.sc_data.metadata.loc[
                embedding_df.index, color_by
            ]

        # create dynamic title
        title = f"UMAP Visualization: by {color_by}"
        # Create plot
        fig = px.scatter(
            embedding_df,
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
        
    def compute_visualize_classifiers(self,X_train,X_test,X_val,y_train,y_test,y_val):
        #Train on classifiers and compare results for UMAP
        #adapted from: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import ListedColormap

        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
        from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.gaussian_process.kernels import RBF
        from sklearn.inspection import DecisionBoundaryDisplay
        from sklearn.naive_bayes import GaussianNB
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        from sklearn.tree import DecisionTreeClassifier

        #set names of classifers for plot
        names = [
            "Nearest Neighbors",
            "Linear SVM",
            "RBF SVM",
            "Decision Tree",
            "Random Forest",
            "Neural Net",
            "AdaBoost",
            "Naive Bayes",
            "QDA",
        ]
        #Gaussian process took forever: 
        #"Gaussian Process",
        #    GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
        #create classifiers
        classifiers = [
            KNeighborsClassifier(3),
            SVC(kernel="linear", C=0.025, random_state=42),
            SVC(gamma=2, C=1, random_state=42),
            DecisionTreeClassifier(max_depth=5, random_state=42),
            RandomForestClassifier(
                max_depth=5, n_estimators=10, max_features=1, random_state=42
            ),
            MLPClassifier(alpha=1, max_iter=1000, random_state=42),
            AdaBoostClassifier(random_state=42),
            GaussianNB(),
            QuadraticDiscriminantAnalysis(),
        ]

        #initialize figure
        figure = plt.figure(figsize=(27, 9))

        #set complete datasets
        #stack 2-d arrays
        X = np.vstack([X_train, X_test, X_val])
        #concatenate 1-d arrays
        y = np.concatenate((y_train, y_test, y_val))

        #set dataset range
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

        cm = plt.cm.RdBu
        cm_bright = ListedColormap(["#FF0000", "#0000FF"])
        ax = plt.subplot(1, len(classifiers) + 1, 1)
        ax.set_title("Input data")

        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
        # Plot the testing points
        ax.scatter(
            X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k"
        )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(())
        ax.set_yticks(())

        # iterate over classifiers
        for i, name in enumerate(names):
            clf = classifiers[i]
            print(f"Initializing classifier: {name}")
            ax = plt.subplot(1, len(classifiers) + 1, i+2)
            #Data needs to be scaled
            clf = make_pipeline(StandardScaler(),clf)
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            DecisionBoundaryDisplay.from_estimator(
                clf, X, cmap=cm, alpha=0.8, ax=ax, eps=0.5
            )

            # Plot the training points
            ax.scatter(
                X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k"
            )
            # Plot the testing points
            ax.scatter(
                X_test[:, 0],
                X_test[:, 1],
                c=y_test,
                cmap=cm_bright,
                edgecolors="k",
                alpha=0.6,
            )

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_title(name)
            ax.text(
                x_max - 0.3,
                y_min + 0.3,
                ("%.2f" % score).lstrip("0"),
                size=15,
                horizontalalignment="right",
            )

    def compare_best_models(self, x_test, y_test):
        #iterate through model names, running _final_test_evaluation(self, x_test, y_test, model_name=None)
        #subplots: test auc, f1 score macro
        
        fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=("Test AUC", "F1-score"))
        x = []
        y1 = []
        y2 = []
        saved_models = self.sc_data.trained_models.keys()
        for model in saved_models:
            results = self.sc_data._final_test_evaluation(x_test, y_test, model_name=model)
            x.append(results['model_name'])
            curr_auc = results['test_auc']
            if curr_auc is None:
                curr_auc=0
            y1.append(curr_auc)
            y2.append(results['f1_score'])
        
        #Sort x,y1,and y2 by values of y1, highest first
        sorted_data = sorted(zip(y1, x, y2), reverse=True)  # Sort by y1 (first element)
        y1, x, y2 = zip(*sorted_data)  # Unpack back into separate lists
        
        # Convert back to lists (zip returns tuples)
        x = list(x)
        y1 = list(y1)
        y2 = list(y2)
        # Calculate shared color range
        all_values = y1 + y2  # Combine both y-value lists
        # exclude 0 val
        all_values = [x for x in all_values if x != 0]
        color_min = min(all_values)
        color_max = max(all_values)

        fig.add_trace(go.Bar(x=x,y=y1,
                             marker={
                                'color': y1, 
                                'colorscale': 'Viridis',
                                'cmin': color_min,
                                'cmax': color_max,
                                'colorbar': dict(
                                    title="Score",
                                    x=1.02,
                                    thickness=15,
                                    len=0.8
                                )
                            }),
                             row=1,col=1)
        fig.add_trace(go.Bar(x=x,y=y2,
                             marker={
                                'color': y2, 
                                'colorscale': 'Viridis',
                                'cmin': color_min,
                                'cmax': color_max,
                                'showscale': False
                            }),
                             row=1,col=2)
        fig.update_layout(
            showlegend=False,
            title_text='Model Comparison for 100 DEGs',
            margin=dict(r=100)  # Add right margin for colorbar
        )
        fig.show()

    
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
        self.sc_data.find_discriminative_genes()
        self.sc_data.run_pca(n_comps=30)

        #Run Harmony and UMAP using generated PCA
        self.sc_data.run_harmony(batch_key="batch")
        self.sc_data.run_umap()

        #Run UMAP on PCA training subset, fit to test/val
        #umap_embedding_trained, umap_embedding_test, umap_embedding_val
        self.sc_data.run_umap(use_subsets=True)

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
    
    def run_ML_pipeline(self,plot_figs=False,compute_classifiers=False):
        # Used AFTER run_pipeline to generate train/test/val subsets of data and visualize PCAs and UMAP generated on subsets
        
        #Generate train/test/val split of overall dataset
        self.sc_data.split_patient_level()

        #Run PCA on train/test/split individually
        # Run PCA on all subsets
        for subset in ["train", "test", "val"]:
            self.sc_data.run_pca(use_split=True, split_subset=subset)

        # Run Harmony on all subsets (if needed)
        for subset in ["train", "test", "val"]:
            self.sc_data.run_harmony(batch_key="batch", subset=subset)

        # Run UMAP with Harmony correction
        self.sc_data.run_umap(use_harmony=True, use_subsets=True)

        # Access the UMAP embeddings
        train_umap = self.sc_data.embedding_df_train
        test_umap = self.sc_data.embedding_df_test
        val_umap = self.sc_data.embedding_df_val

        if plot_figs:
            #Visualize 1st and 2nd PCA results
            self.plot_pca(subset='train', components=[0, 1])
            self.plot_pca(subset='test', components=[0, 1])
            self.plot_pca(subset='val', components=[0, 1])
            #Visualize UMAP results
            self.plot_umap(use_split=True, color_by="traintestsplit")

        if compute_classifiers:
            import numpy as np
            X_train = self.sc_data.umap_embedding_train
            X_test = self.sc_data.umap_embedding_test
            X_val = self.sc_data.umap_embedding_val
            #convert the following from categorical to binary arrays
            y_train = self.sc_data.metadata_train['ADdiag2types'].copy()
            y_train = y_train.map({'nonAD': 0, 'AD': 1})
            y_test = self.sc_data.metadata_test['ADdiag2types'].copy()
            y_test = y_test.map({'nonAD': 0, 'AD': 1})
            y_val = self.sc_data.metadata_val['ADdiag2types'].copy()
            y_val = y_val.map({'nonAD': 0, 'AD': 1})
            #Compute classifiers and visualize, takes 1 min on UMAP data
            self.compute_visualize_classifiers(X_train,X_test,X_val,y_train,y_test,y_val)
        
        print('Run ML pipeline complete.')