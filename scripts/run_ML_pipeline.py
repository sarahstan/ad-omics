"""
run_ML_pipeline: A module using PCA features from sc-RNAseq data 
(specifically, ROSMAP from Sun et al., 2023) to visualize model 
performance in a 2D space. 
Works with scDATA class in scdata.py
Works with scPLOT class in scplot.py
"""  

from data import scDATA
from tools import scPLOT

def run_ML_pipeline(scplot,plot_figs=False,compute_classifiers=False):
    # Used AFTER run_pipeline to generate train/test/val subsets of data and visualize PCAs and UMAP generated on subsets
    
    #Generate train/test/val split of overall dataset
    scplot.sc_data.split_patient_level()

    #Run PCA on train/test/split individually
    # Run PCA on all subsets
    for subset in ["train", "test", "val"]:
        scplot.sc_data.run_pca(use_split=True, split_subset=subset)

    # Run Harmony on all subsets (if needed)
    for subset in ["train", "test", "val"]:
        scplot.sc_data.run_harmony(batch_key="batch", subset=subset)

    # Run UMAP with Harmony correction
    scplot.sc_data.run_umap(use_harmony=True, use_subsets=True)

    # Access the UMAP embeddings, if needed
    # train_umap = scplot.sc_data.embedding_df_train
    # test_umap = scplot.sc_data.embedding_df_test
    # val_umap = scplot.sc_data.embedding_df_val

    if plot_figs:
        #Visualize 1st and 2nd PCA results
        scplot.plot_pca(subset='train', components=[0, 1])
        scplot.plot_pca(subset='test', components=[0, 1])
        scplot.plot_pca(subset='val', components=[0, 1])
        #Visualize UMAP results
        scplot.plot_umap(use_split=True, color_by="traintestsplit")

    if compute_classifiers:
        import numpy as np
        X_train = scplot.sc_data.umap_embedding_train
        X_test = scplot.sc_data.umap_embedding_test
        X_val = scplot.sc_data.umap_embedding_val
        #convert the following from categorical to binary arrays
        y_train = scplot.sc_data.metadata_train['ADdiag2types'].copy()
        y_train = y_train.map({'nonAD': 0, 'AD': 1})
        y_test = scplot.sc_data.metadata_test['ADdiag2types'].copy()
        y_test = y_test.map({'nonAD': 0, 'AD': 1})
        y_val = scplot.sc_data.metadata_val['ADdiag2types'].copy()
        y_val = y_val.map({'nonAD': 0, 'AD': 1})
        
        #Compute classifiers and visualize, takes 1 min on UMAP data
        scplot.compute_visualize_classifiers(X_train,X_test,X_val,y_train,y_test,y_val)
    
    print('Run ML pipeline complete.')