import os
import torch
import lightning as ltn
from torch.utils.data import DataLoader, Subset

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.air import session
from functools import partial
import numpy as np
import json
import mlflow
from ray.air.integrations.mlflow import MLflowLoggerCallback

from trainer.transformer.classifier import ADClassifierLightning
from data import scDATA, TokenData
from data.dataset import ADOmicsDataset
from configs import CellStateEncoderConfig, ScRNATransformerConfig, TrainingConfig


def train_model_with_hyperparams(config, data_path, num_epochs=5, subset_size=500):
    """Training function for transformer hyperparameter search"""

    # Load datasets
    scdata = scDATA(data_path=data_path)
    scdata.split_patient_level()

    # Create token datasets for transformer
    training_dataset = ADOmicsDataset(scDATA=scdata, subset="train", representation="tokens")
    validation_dataset = ADOmicsDataset(scDATA=scdata, subset="val", representation="tokens")

    # Create subsets
    subset_size = min(subset_size, len(training_dataset))
    indices = np.random.choice(len(training_dataset), subset_size, replace=False)
    training_dataset = Subset(training_dataset, indices)

    val_subset_size = min(subset_size // 4, len(validation_dataset))
    val_indices = np.random.choice(len(validation_dataset), val_subset_size, replace=False)
    validation_dataset = Subset(validation_dataset, val_indices)

    # Create dataloaders with TokenData collate
    training_dataloader = DataLoader(
        training_dataset,
        batch_size=config["batch_size"],
        num_workers=4,
        shuffle=True,
        collate_fn=TokenData.collate,
    )

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=config["batch_size"],
        num_workers=4,
        shuffle=False,
        collate_fn=TokenData.collate,
    )

    # Get dataset parameters from the data
    sample = training_dataset.dataset[training_dataset.indices[0]]
    num_genes_total = scdata.adata.n_vars  # Total number of genes in the dataset
    num_cell_types = len(scdata.cell_types)

    # Create configs from hyperparameters
    cell_state_config = CellStateEncoderConfig(
        num_genes_total=num_genes_total,
        gene_embedding_dim=config["embed_dim"],
        num_cell_types=num_cell_types,
        use_film=config.get("use_film", True),
        dropout=config.get("cell_dropout", 0.1),
    )

    transformer_config = ScRNATransformerConfig(
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        ff_dim=config["ff_dim"],
        num_layers=config["num_layers"],
        max_seq_len=512,
        dropout=config.get("transformer_dropout", 0.1),
    )

    training_config = TrainingConfig(
        batch_size=config["batch_size"],
        epochs=num_epochs,
        learning_rate=config["learning_rate"],
        l1_lambda=config["l1_lambda"],
    )

    # Initialize transformer model
    model = ADClassifierLightning(
        cell_state_encoder_config=cell_state_config,
        scrna_transformer_config=transformer_config,
        training_config=training_config,
    )

    # Setup metrics reporting callback
    metrics = {"loss": "val_total_loss"}
    callbacks = [TuneReportCallback(metrics, on="validation_end")]

    trial_dir = session.get_trial_dir()
    logger = ltn.pytorch.loggers.TensorBoardLogger(save_dir=trial_dir)

    # Set up trainer - minimal for quick search
    num_gpus = torch.cuda.device_count()
    trainer = ltn.Trainer(
        max_epochs=num_epochs,
        callbacks=callbacks,
        accelerator="gpu" if num_gpus > 0 else "cpu",
        devices=num_gpus if num_gpus > 0 else None,
        enable_progress_bar=False,
        logger=logger,
    )

    trainer.fit(model, training_dataloader, validation_dataloader)


def get_best_params_from_mlflow(experiment_name, base_log_dir="./logs"):
    """Retrieve the best parameters from a previous MLflow experiment"""
    mlflow_log_dir = os.path.join(base_log_dir, "mlflow")
    mlflow_tracking_uri = f"file:{mlflow_log_dir}"
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    # Get the experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        print(f"No experiment named '{experiment_name}' found. Using default parameters.")
        return None

    # Get all runs in this experiment
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

    if runs.empty:
        print(f"No runs found for experiment '{experiment_name}'. Using default parameters.")
        return None

    # Find the run with the lowest loss
    best_run = runs.sort_values("metrics.loss", ascending=True).iloc[0]

    # Extract parameters
    params = {}
    for key in best_run.filter(like="params").index:
        param_name = key.replace("params.", "")
        param_value = best_run[key]

        # Convert strings to appropriate types
        if param_name in [
            "batch_size",
            "layer_1_size",
            "layer_2_size",
            "layer_3_size",
            "layer_4_size",
        ]:
            param_value = int(param_value)
        elif param_name in ["learning_rate", "l1_lambda"]:
            param_value = float(param_value)

        params[param_name] = param_value

    print(f"Retrieved best parameters from experiment '{experiment_name}':")
    print(json.dumps(params, indent=2))

    return params


def run_hyperparameter_search(
    data_path: str,
    num_samples: int = 20,
    num_epochs: int = 5,
    subset_size: int = 500,
    experiment_name: str = "search",
    coarse_experiment_name: str = None,
    is_fine_tuning: bool = False,
    range_multiplier: float = 0.5,  # Controls search space width for fine-tuning
):
    """Run hyperparameter search and save results to file"""
    # Create organized log directories
    base_log_dir = os.path.abspath("./logs")
    tb_log_dir = os.path.join(base_log_dir, "tensorboard", experiment_name)
    mlflow_log_dir = os.path.join(base_log_dir, "mlflow")

    # Ensure directories exist
    os.makedirs(tb_log_dir, exist_ok=True)
    os.makedirs(mlflow_log_dir, exist_ok=True)

    # Define search space based on search type
    if is_fine_tuning and coarse_experiment_name:
        # Get best parameters from previous experiment
        best_params = get_best_params_from_mlflow(coarse_experiment_name, base_log_dir)

        if best_params:
            # Define a fine-tuned search space around the best parameters
            lr = best_params["learning_rate"]
            l1_lambda = best_params["l1_lambda"]

            config = {
                "batch_size": tune.choice(
                    [
                        max(16, int(best_params["batch_size"] * 0.5)),
                        best_params["batch_size"],
                        min(256, int(best_params["batch_size"] * 2)),
                    ]
                ),
                "learning_rate": tune.loguniform(
                    lr * (1 - range_multiplier), lr * (1 + range_multiplier)
                ),
                "l1_lambda": tune.loguniform(
                    l1_lambda * (1 - range_multiplier), l1_lambda * (1 + range_multiplier)
                ),
            }

            # For each layer size, create a focused search space
            for i in range(1, 5):
                layer_key = f"layer_{i}_size"
                if layer_key in best_params:
                    base_size = best_params[layer_key]
                    smaller = max(5, int(base_size * (1 - range_multiplier)))
                    larger = int(base_size * (1 + range_multiplier))
                    config[layer_key] = tune.choice([smaller, base_size, larger])
        else:
            # Fallback to default search space if no best params found
            config = _create_default_search_space()
    else:
        # Use default search space for coarse-grained search
        config = _create_default_search_space()

    # Define scheduler for early stopping
    scheduler = ASHAScheduler(
        metric="loss", mode="min", max_t=num_epochs, grace_period=2, reduction_factor=2
    )

    # Create partial function with fixed arguments
    train_fn = partial(
        train_model_with_hyperparams,
        data_path=data_path,
        num_epochs=num_epochs,
        subset_size=subset_size,
    )

    # Set up MLflow tracking
    mlflow_tracking_uri = f"file:{mlflow_log_dir}"
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    # MLflow callback for Ray Tune
    mlflow_callback = MLflowLoggerCallback(
        experiment_name=experiment_name, tracking_uri=mlflow_tracking_uri, save_artifact=True
    )

    # Run hyperparameter search
    result = tune.run(
        train_fn,
        resources_per_trial={"cpu": 4, "gpu": 0.5 if torch.cuda.device_count() > 0 else 0},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=tune.CLIReporter(
            parameter_columns=["batch_size", "learning_rate", "l1_lambda"],
            metric_columns=["loss", "training_iteration"],
        ),
        name="adomics_hyperparam_search",
        storage_path=tb_log_dir,  # Use TB directory for Ray Tune's storage
        callbacks=[mlflow_callback],  # Add MLflow callback
    )

    # Get best trial
    best_trial = result.get_best_trial("loss", "min", "last")
    best_config = best_trial.config
    print(f"Best trial config: {best_config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")

    # Save best params to file
    results_dir = os.path.join(base_log_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    best_params_path = os.path.join(results_dir, f"{experiment_name}_best_hyperparams.json")

    with open(best_params_path, "w") as f:
        json.dump(best_config, f, indent=4)

    # Also log best parameters to MLflow
    with mlflow.start_run(run_name="best_trial"):
        mlflow.log_params(best_config)
        mlflow.log_metric("best_loss", best_trial.last_result["loss"])
        mlflow.log_artifact(best_params_path)

    print(f"Best hyperparameters saved to {best_params_path}")
    print(f"TensorBoard logs saved to {tb_log_dir}")
    print(f"MLflow experiment data saved to {mlflow_log_dir}")
    print("To view TensorBoard logs: tensorboard --logdir=./logs/tensorboard")
    print("To view MLflow experiments: mlflow ui --backend-store-uri=file:./logs/mlflow")
    return best_config


def _create_default_search_space():
    """Create a default search space for transformer hyperparameters"""
    return {
        "batch_size": tune.choice([2]),
        "learning_rate": tune.choice([1e-5]),
        "l1_lambda": tune.choice([1e-5]),
        "embed_dim": tune.choice([4]),
        "num_heads": tune.choice([2]),
        "ff_dim": tune.choice([8]),
        "num_layers": tune.choice([2]),
        "use_film": tune.choice([True]),
        "cell_dropout": tune.choice([0.3]),
        "transformer_dropout": tune.choice([0.3]),
    }
    # return {
    #     "batch_size": tune.choice([16, 32, 64]),
    #     "learning_rate": tune.loguniform(1e-5, 1e-2),
    #     "l1_lambda": tune.loguniform(1e-5, 1e-1),
    #     "embed_dim": tune.choice([16, 32, 64]),
    #     "num_heads": tune.choice([2, 4, 8]),
    #     "ff_dim": tune.choice([64, 128, 256]),
    #     "num_layers": tune.choice([2, 4, 6]),
    #     "use_film": tune.choice([True]),
    #     "cell_dropout": tune.uniform(0.0, 0.3),
    #     "transformer_dropout": tune.uniform(0.0, 0.3),
    # }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hyperparameter search for AD Omics Classifier")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/mnt/c/Users/JJ/Dropbox/Sharejerah/ROSMAP/data",
        help="Path to data directory",
    )
    parser.add_argument(
        "--num_trials", type=int, default=20, help="Number of hyperparameter combinations to try"
    )
    parser.add_argument(
        "--subset_size", type=int, default=500, help="Number of examples to use for quick search"
    )
    parser.add_argument("--epochs", type=int, default=5, help="Maximum epochs per trial")
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="adomics_hp_search",
        help="Name of the experiment for tracking",
    )
    parser.add_argument(
        "--fine_tuning",
        action="store_true",
        help="Whether to perform fine-tuning around best parameters from a previous experiment",
    )
    parser.add_argument(
        "--coarse_experiment_name",
        type=str,
        default=None,
        help="Name of the coarse-grained experiment to use for fine-tuning",
    )
    parser.add_argument(
        "--range_multiplier",
        type=float,
        default=0.5,
        help="Range multiplier for fine-tuning search space (0.5 means +/- 50%)",
    )

    args = parser.parse_args()

    run_hyperparameter_search(
        data_path=args.data_path,
        num_samples=args.num_trials,
        num_epochs=args.epochs,
        subset_size=args.subset_size,
        experiment_name=args.experiment_name,
        coarse_experiment_name=args.coarse_experiment_name,
        is_fine_tuning=args.fine_tuning,
        range_multiplier=args.range_multiplier,
    )
