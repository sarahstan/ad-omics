import lightning as ltn
import json
import os
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from trainer.lightning.mlp_classifier import ADClassifierLightning
from data import scDATA, ADOmicsDataset


def load_hyperparameters(json_path="best_hyperparams.json"):
    """Load hyperparameters from JSON file"""

    default_params = {
        "batch_size": 8,
        "learning_rate": 0.001,
        "l1_lambda": 0.0,
        "hidden_dims": [5000, 500, 50, 5],
    }

    if not os.path.exists(json_path):
        print(f"Warning: {json_path} not found. Using default hyperparameters.")
        return default_params

    try:
        with open(json_path, "r") as f:
            hyperparams = json.load(f)

        # Extract layer sizes into hidden_dims list
        hidden_dims = []
        for i in range(1, 5):
            layer_key = f"layer_{i}_size"
            if layer_key in hyperparams and hyperparams[layer_key] > 0:
                hidden_dims.append(hyperparams[layer_key])

        # Add hidden_dims to the hyperparams dict
        hyperparams["hidden_dims"] = hidden_dims

        print(f"Loaded hyperparameters from {json_path}: {hyperparams}")
        return hyperparams
    except Exception as e:
        print(f"Error loading hyperparameters: {e}. Using default values.")
        return default_params


def main():
    # Load hyperparameters
    hyperparams = load_hyperparameters()

    ### Load datasets
    data_path = "/mnt/c/Users/JJ/Dropbox/Sharejerah/ROSMAP/data"
    scdata = scDATA(data_path=data_path)
    scdata.split_patient_level()
    training_dataset = ADOmicsDataset(scDATA=scdata, subset="train")
    training_dataloader = DataLoader(
        training_dataset,
        batch_size=hyperparams["batch_size"],
        num_workers=31,
        shuffle=True,
    )
    validation_dataset = ADOmicsDataset(scDATA=scdata, subset="val")
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=hyperparams["batch_size"],
        num_workers=31,
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
        hidden_dims=hyperparams["hidden_dims"],
        learning_rate=hyperparams["learning_rate"],
        l1_lambda=hyperparams["l1_lambda"],
    )

    ### Set up loggers
    tb_logger = TensorBoardLogger(save_dir="tblogs", name="adomics_mlp")

    ### Set up trainer
    trainer = ltn.Trainer(logger=tb_logger, max_epochs=10)
    trainer.fit(model, training_dataloader, validation_dataloader)


if __name__ == "__main__":
    main()
