import lightning as ltn
import json
import os
import argparse
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from trainer.lightning.mlp.classifier import ADClassifierLightning
from data import scDATA, ADOmicsDataset


def load_hyperparameters(json_path=None, experiment_name=None):
    """Load hyperparameters from JSON file or experiment results"""

    default_params = {
        "batch_size": 8,
        "learning_rate": 0.001,
        "l1_lambda": 0.0,
        "hidden_dims": [5000, 500, 50, 5],
    }

    # If experiment_name is provided, look for its best hyperparams file
    if experiment_name:
        results_dir = os.path.abspath("./logs/results")
        json_path = os.path.join(results_dir, f"{experiment_name}_best_hyperparams.json")
        print(f"Looking for hyperparameters from experiment '{experiment_name}' at: {json_path}")

    # If no specific path is provided, try to find the most recent best_hyperparams file
    if not json_path:
        results_dir = os.path.abspath("./logs/results")
        if os.path.exists(results_dir):
            # List all hyperparameter JSON files and sort by modification time (newest first)
            hp_files = [f for f in os.listdir(results_dir) if f.endswith("_best_hyperparams.json")]
            if hp_files:
                hp_files.sort(
                    key=lambda x: os.path.getmtime(os.path.join(results_dir, x)), reverse=True
                )
                json_path = os.path.join(results_dir, hp_files[0])
                print(f"Using most recent hyperparameters file: {json_path}")
            else:
                print("No hyperparameter files found in results directory.")
                return default_params
        else:
            # Try the root directory as fallback
            if os.path.exists("best_hyperparams.json"):
                json_path = "best_hyperparams.json"
            else:
                print("No hyperparameter files found. Using default hyperparameters.")
                return default_params

    # Load hyperparameters from the JSON file
    if not os.path.exists(json_path):
        print(f"Warning: {json_path} not found. Using default hyperparameters.")
        return default_params

    try:
        with open(json_path, "r") as f:
            hyperparams = json.load(f)

        # Extract layer sizes into hidden_dims list
        hidden_dims = []
        for i in range(1, 10):  # Support up to 9 layers for future flexibility
            layer_key = f"layer_{i}_size"
            if layer_key in hyperparams and hyperparams[layer_key] > 0:
                hidden_dims.append(hyperparams[layer_key])

        # Add hidden_dims to the hyperparams dict
        hyperparams["hidden_dims"] = hidden_dims

        print(f"Loaded hyperparameters from {json_path}:")
        print(f"  Batch size: {hyperparams.get('batch_size', default_params['batch_size'])}")
        print(
            f"  Learning rate: {hyperparams.get('learning_rate', default_params['learning_rate'])}"
        )
        print(f"  L1 lambda: {hyperparams.get('l1_lambda', default_params['l1_lambda'])}")
        print(f"  Network architecture: {hidden_dims}")

        return hyperparams
    except Exception as e:
        print(f"Error loading hyperparameters: {e}. Using default values.")
        return default_params


def main(args):
    # Load hyperparameters from specified experiment or file
    hyperparams = load_hyperparameters(args.param_file, args.experiment_name)

    ### Load datasets
    data_path = args.data_path
    scdata = scDATA(data_path=data_path)
    scdata.split_patient_level()

    training_dataset = ADOmicsDataset(scDATA=scdata, subset="train")
    training_dataloader = DataLoader(
        training_dataset,
        batch_size=hyperparams.get("batch_size", 8),
        num_workers=args.num_workers,
        shuffle=True,
    )

    validation_dataset = ADOmicsDataset(scDATA=scdata, subset="val")
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=hyperparams.get("batch_size", 8),
        num_workers=args.num_workers,
        shuffle=False,
    )

    ### Get example size
    gene_count_vector_input_size = training_dataset[0][0].shape[0]

    ### Cell type input size
    cell_type_input_size = training_dataset[0][1].shape[0]

    ### Initialize model
    model = ADClassifierLightning(
        gene_input_dim=gene_count_vector_input_size,
        cell_type_input_dim=cell_type_input_size,
        hidden_dims=hyperparams.get("hidden_dims", [5000, 500, 50, 5]),
        learning_rate=hyperparams.get("learning_rate", 0.001),
        l1_lambda=hyperparams.get("l1_lambda", 0.0),
    )

    ### Set up loggers
    log_dir = os.path.join("logs", "tensorboard") if args.use_log_dir else "tblogs"
    tb_logger = TensorBoardLogger(save_dir=log_dir, name=args.experiment_name or "adomics_mlp")

    ### Set up trainer
    trainer = ltn.Trainer(
        logger=tb_logger,
        max_epochs=args.epochs,
        accelerator="gpu" if args.use_gpu else "cpu",
        devices=1 if args.use_gpu else None,
    )

    trainer.fit(model, training_dataloader, validation_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train AD Omics Classifier with optimized hyperparameters"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="/mnt/c/Users/JJ/Dropbox/Sharejerah/ROSMAP/data",
        help="Path to data directory",
    )
    parser.add_argument(
        "--param_file",
        type=str,
        default=None,
        help="Path to hyperparameter JSON file",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Name of hyperparameter experiment to use (will look for best params file)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=31,
        help="Number of dataloader workers",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Use GPU for training if available",
    )
    parser.add_argument(
        "--use_log_dir",
        action="store_true",
        help="Use the ./logs/tensorboard directory structure for logs",
    )

    args = parser.parse_args()
    main(args)
