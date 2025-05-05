import os
from typing import Tuple
import torch
import lightning as ltn
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from models.mlp_classifier import ADClassifier
from data import ADOmicsDataset


class ADOmicsMLPClassifier(ltn.LightningModule):
    def __init__(self, input_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.model = ADClassifier(input_dim=input_dim, hidden_dims=hidden_dims)
        self.loss_fn = self.model.loss_fn

    def forward(self, x):
        return self.model(x)

    def _get_loss(
        self, batch: Tuple[torch.Tensor, torch.Tensor], prefix: str = "", write_to_log: bool = True
    ):
        x, y = batch
        outputs = self(x).reshape(-1)
        loss = self.loss_fn(outputs, y)
        if write_to_log:
            name = "loss"
            if prefix:
                name = f"{prefix}_" + name
            self.log(name, loss)  # This will log the loss automatically
        return loss

    def training_step(self, batch, batch_idx):
        return self._get_loss(batch)

    def validation_step(self, batch, batch_idx):
        return self._get_loss(batch, prefix="val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def on_train_epoch_end(self):
        logger = self.logger.experiment  # TensorBoard SummaryWriter
        for name, param in self.model.named_parameters():
            # Log weights
            logger.add_histogram(f"weights/{name}", param, self.current_epoch)

            # Log gradients, if available
            if param.grad is not None:
                logger.add_histogram(f"gradients/{name}", param.grad, self.current_epoch)


def main():
    ### Load datasets
    data_path = "/mnt/c/Users/JJ/Dropbox/Sharejerah/ROSMAP/data"
    training_dataset = ADOmicsDataset(data_path=data_path, subset="train")
    training_dataloader = DataLoader(training_dataset, batch_size=8, num_workers=31)
    validation_dataset = ADOmicsDataset(data_path=data_path, subset="validation")
    validation_dataloader = DataLoader(validation_dataset, batch_size=2, num_workers=31)

    ### Get example size
    example_size = training_dataset[0][0].shape[0]

    ### Initialize model
    model = ADOmicsMLPClassifier(input_dim=example_size, hidden_dims=[5000, 500, 50, 5])

    ### Set up loggers
    tb_logger = TensorBoardLogger(save_dir="tblogs", name="adomics_mlp")

    ### Set up trainer
    trainer = ltn.Trainer(logger=tb_logger, max_epochs=10)
    trainer.fit(model, training_dataloader, validation_dataloader)


if __name__ == "__main__":
    main()
