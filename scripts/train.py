from trainer import Trainer
from models.mlp_classifier import ADClassifier


def main():
    training_loader = 0
    validation_loader = 0
    model = ADClassifier(input_dim=0, hidden_dims=[5000, 500, 50, 5])

    trainer = Trainer(
        training_loader=training_loader,
        validation_loader=validation_loader,
        model=model,
    )

    trainer.train()


if __name__ == "__main__":
    main()
