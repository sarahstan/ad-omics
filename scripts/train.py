from trainer import Trainer
from models.mlp_classifier import ADClassifier
from data import ADOmicsDataLoader


def main():
    data_path = "/mnt/c/Users/JJ/Dropbox/Sharejerah/ROSMAP/data"
    addl_train = ADOmicsDataLoader(data_path=data_path, subset="train")
    training_loader = addl_train.get_dataloader()
    addl_validattion = ADOmicsDataLoader(data_path=data_path, subset="train")
    validation_loader = addl_validattion.get_dataloader()
    model = ADClassifier(input_dim=0, hidden_dims=[5000, 500, 50, 5])

    trainer = Trainer(
        training_loader=training_loader,
        validation_loader=validation_loader,
        model=model,
    )

    trainer.train()


if __name__ == "__main__":
    main()
