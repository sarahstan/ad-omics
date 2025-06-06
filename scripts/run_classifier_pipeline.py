"""
run_classifier_pipeline: A module for using differentially-expressed genes in an adata object
to train ML classifiers for AD disease outcomes. Works with scDATA and scPLOT types. 
"""     
from data import scDATA
from tools import scPLOT
import numpy as np


def run_classifier_pipeline(scdata,limit_dg=True):
    """Takes full AnnData or HVG subset, adds features to .obsm, splits data and metadata, and fits 
    sequence of classifiers, using five-fold cross-validation and grid search strategies for 
    hyperparameter tuning. Displays AUC values by method, overall and separated by cell type. """

    #Note: will only run DNN on the full dataset, will take significant resources

    #First, add classifier features to the .obsm section
    scdata._add_classifier_features(limit_dg)

    #Second, generate train/test/val subsets for classifiers
    # First, splits
    train_mask, val_mask, test_mask = scdata.split_patient_level()
    x_train, x_val, x_test, y_train, y_val, y_test = scdata._generate_subsets_for_classifier(
        limit_dg,train_mask, val_mask, test_mask)


    #List of classifiers from ScATT paper: Logistic Regression, SVM, Decision Tree, 
    # ADAboost, XGboost, ResNet (residual netork), DNN (deep neural network)
    #Suggested by Claude: RF (random forest), SVM, Logistic Regression. Excluding ResNet due to compute. 
    
    # Run Grid Search with Crossfold Validation to create the best models of training data
    # Save results into self.trained_models[model_name]
    scdata._model_comparison(x_train, x_val, x_test, y_train, y_val, y_test, include_dnn=False, cv_folds=5)

    # Evaluate all models with their best parameters
    scdata._final_test_evaluation()

def _add_classifier_features(scdata,limit_dg):
    #Add helpful classifier features from metadata, including:
    #  celltype, msex. Can add brain_region or others later
    from sklearn.preprocessing import OneHotEncoder
    
    feature_list = ['celltype','msex', 'brain_region'] 

    # One-hot encode
    encoder = OneHotEncoder(sparse_output=False, dtype=int)

    # Encode each feature and add to the appropriate spot
    for feature in feature_list:
        curr_feature = scdata.metadata[feature].values.reshape(-1,1)
        print(curr_feature.shape)
        curr_onehot = encoder.fit_transform(curr_feature)
        print(curr_onehot.shape)
        if limit_dg:
            print(f"Adding X_onehot_{feature} to adata_dg.obsm")
            scdata.adata_dg.obsm[f"X_onehot_{feature}"] = curr_onehot
        else:
            print(f"Adding X_onehot_{feature} to adata.obsm")
            scdata.adata.obsm[f"X_onehot_{feature}"] = curr_onehot

def _generate_subsets_for_classifier(scdata,limit_dg,train_mask, val_mask, test_mask):        

    #Split data and compute subsets for train/test/val        
    if limit_dg:
        # Apply to adata_dg, separate X for train/val/test at end
        x_complete_adata = scdata.adata_dg.copy()
    else: 
        x_complete_adata = scdata.adata.copy()

    #Make train/val/test subsets of y data using masks        
    y_train = scdata.metadata.loc[train_mask[train_mask].index]   
    y_val = scdata.metadata.loc[val_mask[val_mask].index]       
    y_test = scdata.metadata.loc[test_mask[test_mask].index]
    
    # Convert categorical to binary
    y_train = y_train['ADdiag2types'].copy()
    y_train = y_train.map({'nonAD': 0, 'AD': 1})
    y_val = y_val['ADdiag2types'].copy()
    y_val = y_val.map({'nonAD': 0, 'AD': 1})
    y_test = y_test['ADdiag2types'].copy()
    y_test = y_test.map({'nonAD': 0, 'AD': 1})

    #Combine X with obsm classifier features
    if hasattr(x_complete_adata.X, 'toarray'):
        X_expr = x_complete_adata.X.toarray() # Convert sparse to dense
    else: 
        X_expr = x_complete_adata.X
    
    #Get classifier features from .obsm
    classifier_features = []
    obsm_keys = [key for key in x_complete_adata.obsm.keys() if 'X_' in key]
    
    if obsm_keys:
        print(f"Found classifier features in .obsm: {obsm_keys}")
        for key in obsm_keys:
            curr_feature = x_complete_adata.obsm[key]
            print(f"  {key}: {curr_feature.shape}")
            classifier_features.append(curr_feature)
        
        classifier_features_combined = np.hstack(classifier_features)
        X_total_combined = np.hstack([X_expr,classifier_features_combined])
        print(f"Combined features: {X_expr.shape} + {classifier_features_combined.shape} = {X_total_combined.shape}")
    else:
        print("No classifier features found in .obsm, using only .X")
        X_total_combined = X_expr

    #Separate X_total_combined using train/test/val split
    
    #Generate numeric indices boolean for use on arrays
    train_mask_numeric = np.where(train_mask.values)[0]
    val_mask_numeric = np.where(val_mask.values)[0]
    test_mask_numeric = np.where(test_mask.values)[0]
    

    #Make the split
    x_train = X_total_combined[train_mask_numeric]
    x_val = X_total_combined[val_mask_numeric]
    x_test = X_total_combined[test_mask_numeric]

    print(f"Final feature matrices:")
    print(f"  X_train: {x_train.shape}, dtype: {x_train.dtype}")
    print(f"  Y_train: {y_train.shape}, dtype: {y_train.dtype}")
    print(f"  X_val: {x_val.shape}, dtype: {x_val.dtype}")
    print(f"  X_test: {x_test.shape}, dtype: {x_test.dtype}")
    
    
    return x_train, x_val, x_test, y_train, y_val, y_test

def _model_comparison(scdata, x_train, x_val, x_test, y_train, y_val, y_test, 
                        include_dnn=False, cv_folds=5, checkpoint_file='model_checkpoint.pkl'):
    """
    Comprehensive model comparison including all models from the paper.
        Not including ResNet for computational reasons.
        Model List: Logistic Regression, SVM, Decision Tree, Random Forest, ADAboost, XGBoost, optionally DNN
    """
    from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import roc_auc_score
    import xgboost as xgb
    import pandas as pd
    import numpy as np
    import time
    import pickle
    import os

    print(f"Training set, X: {x_train.shape}")
    print(f"Training set, y: {y_train.shape}")
    print(f"Validation set, X: {x_val.shape}")
    print(f"Validation set, y: {y_val.shape}")
    print(f"Test set, X: {x_test.shape}")
    print(f"Test set, y: {y_test.shape}")    

    #Base list of models
    models = {
            'logistic': LogisticRegression(random_state=42, max_iter=1000),
                'svm': SVC(random_state=42),
            'decision_tree': DecisionTreeClassifier(random_state=42),
            'random_forest': RandomForestClassifier(random_state=42),
            'adaboost': AdaBoostClassifier(random_state=42),
            'xgboost': xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        }
    #Base grid parameter list
    param_grids = {
        'logistic': [
            # Focus on l1 penalty with liblinear (your best combination after bigger search)
            {
                'penalty': ['l1'],
                'solver': ['liblinear'],
                'C': [0.01, 0.05, 0.1, 0.2, 0.5],  # Fine-tune around 0.1
                'max_iter': [1000]
            },
            
            # Add a small grid for l2 just to verify
            {
                'penalty': ['l2'],
                'solver': ['liblinear', 'saga'],
                'C': [0.1, 1.0, 10.0],
                'max_iter': [1000]
            },
            
            # Minimal elasticnet option
            {
                'penalty': ['elasticnet'],
                'solver': ['saga'],
                'C': [0.1, 1.0],
                'l1_ratio': [0.5, 0.9],
                'max_iter': [1000]
            }
        ],
        'svm': [
            # Start with just linear kernel - fastest option
            {
                'kernel': ['linear'],
                'C': [1, 10, 100],  # Reduced from 4 to 3 values
            },
            # Only try RBF with very few parameters
            {
                'kernel': ['rbf'],
                'C': [1, 10],       # Only 2 C values
                'gamma': ['scale'], # Only 1 gamma value
            }
        ],
        'decision_tree': {
            # More data = deeper trees allowed
            'max_depth': [10, 15, 20, 25, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 5, 10],
            'criterion': ['gini', 'entropy'],
            'max_features': ['sqrt', 'log2', None]
        },
        'random_forest': {
            # More data = deeper trees allowed
            # ran slowly so reduced complexity
            'n_estimators': [100, 200],
            'max_depth': [10, None],
            'min_samples_split': [2, 10],
            'min_samples_leaf': [1, 5],
            'max_features': ['sqrt']
        },
        'adaboost': {
            # More data = more estimators viable
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.5, 1.0],
            'algorithm': ['SAMME.R']
        },
        'xgboost': {
            # Simplifying to speed things
            'n_estimators': [100, 300], 
            'max_depth': [3, 6],
            'learning_rate': [0.1, 0.3], 
            'subsample': [0.8], 
            'colsample_bytree': [0.8], 
            'reg_alpha': [0], 
            'reg_lambda': [1], 
            'min_child_weight': [1] 
        }
    }

    #To include DNN
    if include_dnn:
        models['dnn'] = MLPClassifier(
            random_state=42, 
            early_stopping=True,
            validation_fraction=0.1,  # Use 10% of training data for early stopping
            n_iter_no_change=10  # Stop if no improvement for 10 epochs
        )
        param_grids['dnn'] = {
            # Simplified architecture options
            'hidden_layer_sizes': [
                (100,),                # Simple single-layer
                (200, 100),            # Medium two-layer
                (300, 150, 75)         # One deeper option
            ],
            'activation': ['relu'],   
            'alpha': [0.0001, 0.001],  # Reduced regularization options
            'learning_rate_init': [0.001, 0.01],  # Most common learning rates
            'batch_size': [64, 128],   # Reasonable batch sizes
            'max_iter': [300]          
        }
    
    # Load previous results or create empty variable
    # Try to load existing results
    results = {}
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'rb') as f:
                results = pickle.load(f)
            print(f"Loaded checkpoint with {len(results)} completed models: {list(results.keys())}")
        except Exception as e:
            print(f"Could not load checkpoint: {e}")
            results = {}

    # Train each model
    for model_name in models.keys():
        # Skip if already completed
        if model_name in results:
            print(f"Skipping {model_name} - already completed")
            continue
        print(f"\n{'='*50}")
        print(f"Training {model_name.upper()}")
        print(f"{'='*50}")
        
        start_time = time.time()

        # Stratified CV for consistent evaluation
        if model_name == 'svm' or model_name == 'random_forest' or model_name == 'xgboost':
            cv_folds_model = 3  # Use 3-fold instead of 5-fold
            print(f"Using {cv_folds_model}-fold CV for {model_name} (faster)")
        else:
            cv_folds_model = cv_folds
        cv_strategy = StratifiedKFold(n_splits=cv_folds_model, shuffle=True, random_state=42)
        
        # Grid search with CV
        grid_search = GridSearchCV(
            estimator=models[model_name],
            param_grid=param_grids[model_name],
            cv=cv_strategy,
            error_score=float('-inf'),
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        # SVM is struggling with the full dataset
        if model_name == 'svm' or model_name == 'random_forest' or model_name == 'xgboost':
            # Use only 50% of training data for SVM, random forest, xgboost
            from sklearn.model_selection import train_test_split
            x_train_subset, _, y_train_subset, _ = train_test_split(
                x_train, y_train, train_size=0.5, random_state=42, stratify=y_train
            )
            print(f"Using subset for {model_name}: {x_train_subset.shape[0]} samples instead of {x_train.shape[0]}")
            grid_search.fit(x_train_subset, y_train_subset)
        else:
            # Fit on training data, other models
            grid_search.fit(x_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_

        # After grid search completes
        print("\n" + "-"*40)
        print(f"BEST MODEL SUMMARY FOR {model_name.upper()}")
        print("-"*40)
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        print(f"Model type: {type(grid_search.best_estimator_).__name__}")

        # Better error handling for predict_proba
        if hasattr(best_model, 'predict_proba'):
            try:
                val_proba = best_model.predict_proba(x_val)[:, 1]
                print(f"Prediction probabilities shape: {val_proba.shape}")
            except Exception as e:
                print(f"Error in predict_proba: {str(e)}")
                val_proba = None
        else:
            print("Model does not have predict_proba method")
            val_proba = None

        try:
            val_accuracy = best_model.score(x_val, y_val)
            print(f"Validation accuracy: {val_accuracy:.4f}")
        except Exception as e:
            print(f"Error calculating validation accuracy: {str(e)}")
            val_accuracy = None

        # Calculate additional metrics with better error handling
        try:
            if val_proba is not None:
                print(f"y_val unique values: {np.unique(y_val)}")
                print(f"val_proba range: [{min(val_proba)}, {max(val_proba)}]")
                val_auc = roc_auc_score(y_val, val_proba)
                print(f"Validation AUC: {val_auc:.4f}")
            else:
                val_auc = None
        except Exception as e:
            print(f"Error calculating AUC: {str(e)}")
            val_auc = None
        
        training_time = time.time() - start_time
        
        # Store results
        results[model_name] = {
            'best_model': best_model,
            'best_params': grid_search.best_params_,
            'cv_score': grid_search.best_score_,
            'val_accuracy': val_accuracy,
            'val_auc': val_auc,
            'training_time': training_time,
            'grid_search': grid_search
        }
        
        #Save results for each model as it completes
        # Save checkpoint after each model
        try:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(results, f)
            print(f"Checkpoint saved with {len(results)} completed models")
        except Exception as e:
            print(f"Could not save checkpoint: {e}")

        print(f"Best CV score: {grid_search.best_score_:.4f}")
        print(f"Validation accuracy: {val_accuracy:.4f}")
        if val_auc:
            print(f"Validation AUC: {val_auc:.4f}")
        print(f"Training time: {training_time:.2f} seconds")
        print(f"Best parameters: {grid_search.best_params_}")
    
    # Create comparison summary
    comparison_df = pd.DataFrame({
        model: {
            'CV_Score': results[model]['cv_score'],
            'Val_Accuracy': results[model]['val_accuracy'],
            'Val_AUC': results[model]['val_auc'],
            'Training_Time': results[model]['training_time']
        }
        for model in results.keys()
    }).T
    
    # Sort by validation AUC, better for RNAseq data
    comparison_df.sort_values('Val_AUC', ascending=False, inplace=True)
    
    print(f"\n{'='*60}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(comparison_df)
    
    # Save all results to self for later access
    scdata.model_comparison_results = results
    scdata.trained_models = {name: results[name]['best_model'] for name in results.keys()}
    scdata.best_parameters = {name: results[name]['best_params'] for name in results.keys()}
    
    # Find overall best model
    best_model_name = comparison_df.index[0]
    scdata.best_model_name = best_model_name
    scdata.best_model = results[best_model_name]['best_model']
    
    print(f"\nOverall best model: {best_model_name}")
    print(f"Best validation AUC: {comparison_df.loc[best_model_name, 'Val_AUC']:.4f}")
    print(f"\nAll trained models available: {list(scdata.trained_models.keys())}")
    
    return results, comparison_df

def _final_test_evaluation(scdata, x_test, y_test, model_name=None):
    """
    Final evaluation on test set with specified model or best model
    
    Parameters:
    model_name (str): Name of model to evaluate ('logistic', 'xgboost', etc.)
                    If None, uses the overall best model from comparison

    model list: 'logistic','svm','decision_tree','random_forest','adaboost','xgboost','dnn'

    """
    from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, f1_score

    if not hasattr(scdata, 'trained_models'):
        raise ValueError("Run _model_comparison() first!")
    
    # Select model to evaluate
    if model_name is None:
        if not hasattr(scdata, 'best_model_name'):
            raise ValueError("No best model found. Run _model_comparison() first!")
        model_name = scdata.best_model_name
        selected_model = scdata.best_model
        scdata._print(f"Using overall best model: {model_name}")
    else:
        if model_name not in scdata.trained_models:
            available_models = list(scdata.trained_models.keys())
            raise ValueError(f"Model '{model_name}' not found. Available models: {available_models}")
        selected_model = scdata.trained_models[model_name]
        scdata._print(f"Using specified model: {model_name}")
    
    # Final predictions
    test_pred = selected_model.predict(x_test)
    test_proba = selected_model.predict_proba(x_test)[:, 1] if hasattr(selected_model, 'predict_proba') else None
    test_accuracy = selected_model.score(x_test, y_test)
    
    try:
        test_auc = roc_auc_score(y_test, test_proba) if test_proba is not None else None
    except:
        test_auc = None
    
    scdata._print(f"\n{'='*50}")
    scdata._print(f"FINAL TEST EVALUATION - {model_name.upper()}")
    scdata._print(f"{'='*50}")
    scdata._print(f"Best parameters: {scdata.best_parameters[model_name]}")
    scdata._print(f"Test Accuracy: {test_accuracy:.4f}")
    if test_auc:
        scdata._print(f"Test AUC: {test_auc:.4f}")
    
    scdata._print("\nClassification Report:")
    scdata._print(classification_report(y_test, test_pred))
    
    scdata._print("\nConfusion Matrix:")
    scdata._print(confusion_matrix(y_test, test_pred))
    
    return {
        'model_name': model_name,
        'best_parameters': scdata.best_parameters[model_name],
        'test_accuracy': test_accuracy,
        'test_auc': test_auc,
        'f1_score': f1_score(y_test, test_pred, average='macro'),
        'predictions': test_pred,
        'probabilities': test_proba
    }

def compare_best_models(self, x_test, y_test):
        #iterate through model names, running _final_test_evaluation(self, x_test, y_test, model_name=None)
        #subplots: test auc, f1 score macro
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

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