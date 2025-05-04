from trainer import Trainer
from models.mlp_classifier import ADClassifier
from data import ADOmicsDataset
from torch.utils.data import DataLoader


def main():
    data_path = "/mnt/c/Users/JJ/Dropbox/Sharejerah/ROSMAP/data"
    training_dataset = ADOmicsDataset(data_path=data_path, subset="train")
    training_dataloader = DataLoader(training_dataset, batch_size=2)
    breakpoint()
    validation_dataset = ADOmicsDataset(data_path=data_path, subset="validation")
    validation_dataloader = DataLoader(validation_dataset, batch_size=2)
    example_size = training_dataset[0][0].shape[0]
    model = ADClassifier(input_dim=example_size, hidden_dims=[5000, 500, 50, 5])

    trainer = Trainer(
        training_loader=training_dataloader,
        validation_loader=validation_dataloader,
        model=model,
    )

    trainer.train()


if __name__ == "__main__":
    main()
