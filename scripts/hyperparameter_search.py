import torch
import lightning as ltn
from torch.utils.data import DataLoader, Subset
from models.lightning.mlp_classifier import ADClassifierLightning
from data import scDATA, ADOmicsDataset
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from functools import partial
import numpy as np
import json


def train_model_with_hyperparams(config, data_path, num_epochs=5, subset_size=500):
    """Training function for hyperparameter search"""
    # Load datasets
    scdata = scDATA(data_path=data_path)
    scdata.split_patient_level()

    # Create datasets
    training_dataset = ADOmicsDataset(scDATA=scdata, subset="train")
    validation_dataset = ADOmicsDataset(scDATA=scdata, subset="val")

    # Create subset for quick hyperparameter search
    subset_size = min(subset_size, len(training_dataset))
    indices = np.random.choice(len(training_dataset), subset_size, replace=False)
    training_dataset = Subset(training_dataset, indices)

    # Smaller validation subset
    val_subset_size = min(subset_size // 4, len(validation_dataset))
    val_indices = np.random.choice(len(validation_dataset), val_subset_size, replace=False)
    validation_dataset = Subset(validation_dataset, val_indices)

    # Create dataloaders
    training_dataloader = DataLoader(
        training_dataset,
        batch_size=config["batch_size"],
        num_workers=4,  # Reduced workers for quick runs
    )

    validation_dataloader = DataLoader(
        validation_dataset, batch_size=config["batch_size"], num_workers=4
    )

    # Get input dimensions from the first sample
    sample = training_dataset.dataset[training_dataset.indices[0]]
    gene_count_vector_input_size = sample[0].shape[0]
    cell_type_input_size = sample[1].shape[0]

    # Parse hidden dimensions from config
    hidden_dims = []
    for i in range(1, 5):
        layer_size_key = f"layer_{i}_size"
        if layer_size_key in config and config[layer_size_key] > 0:
            hidden_dims.append(config[layer_size_key])

    # Initialize model
    model = ADClassifierLightning(
        gene_input_dim=gene_count_vector_input_size,
        cell_type_input_dim=cell_type_input_size,
        hidden_dims=hidden_dims,
        learning_rate=config["learning_rate"],
        l1_lambda=config["l1_lambda"],
    )

    # Setup metrics reporting callback
    metrics = {"loss": "val_total_loss"}
    callbacks = [TuneReportCallback(metrics, on="validation_end")]

    # Set up trainer - minimal for quick search
    num_gpus = torch.cuda.device_count()
    trainer = ltn.Trainer(
        max_epochs=num_epochs,
        callbacks=callbacks,
        accelerator="gpu" if num_gpus > 0 else "cpu",
        devices=num_gpus if num_gpus > 0 else None,
        enable_progress_bar=False,
        logger=False,  # No logging needed for quick search
    )

    trainer.fit(model, training_dataloader, validation_dataloader)


def run_hyperparameter_search(data_path, num_samples=20, num_epochs=5, subset_size=500):
    """Run hyperparameter search and save results to file"""
    # Define search space
    config = {
        "batch_size": tune.choice([8, 16, 32, 64]),
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "l1_lambda": tune.loguniform(1e-5, 1e-3),
        "layer_1_size": tune.choice([2000, 3000, 5000, 7000]),
        "layer_2_size": tune.choice([250, 500, 750]),
        "layer_3_size": tune.choice([30, 50, 100]),
        "layer_4_size": tune.choice([5, 10, 20]),
    }

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
        local_dir="./ray_results",
    )

    # Get best trial
    best_trial = result.get_best_trial("loss", "min", "last")
    best_config = best_trial.config
    print(f"Best trial config: {best_config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")

    # Save best params to file
    with open("best_hyperparams.json", "w") as f:
        json.dump(best_config, f, indent=4)

    print("Best hyperparameters saved to best_hyperparams.json")
    return best_config


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

    args = parser.parse_args()

    run_hyperparameter_search(
        data_path=args.data_path,
        num_samples=args.num_trials,
        num_epochs=args.epochs,
        subset_size=args.subset_size,
    )
