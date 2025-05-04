from datetime import datetime
import torch
from typing import Optional
from torch.optim import Adam
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        training_loader: DataLoader,
        validation_loader: DataLoader,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[torch.device] = None,
    ):
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.model = model.to(self.device)
        if optimizer is None:
            self.optimizer = Adam(model.parameters())
        else:
            self.optimizer = optimizer

    def train_one_epoch(self, epoch_index: int):
        running_loss = 0.0
        total_batches = 0
        epoch_average_loss = 0.0

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(self.training_loader):
            # Every data instance is an input + label pair
            inputs, labels = data

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self.model(inputs).reshape(-1)

            # Compute the loss and its gradients
            loss = self.model.loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            total_batches += 1

            if i % 10 == 0:
                epoch_average_loss = running_loss / total_batches  # loss per batch
                tb_x = epoch_index * len(self.training_loader) + i + 1
                print(f"Epoch: {epoch_index}")
                print(f"Training batch: {tb_x}")
                print(f"Loss: {epoch_average_loss:0.4e}")

        return epoch_average_loss

    def train(self, total_epochs: int = 5):
        # Initializing in a separate cell so we can easily add more epochs to the same run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # writer = SummaryWriter("runs/fashion_trainer_{}".format(timestamp))
        epoch_number = 0

        best_vloss = 1_000_000.0

        for epoch in range(total_epochs):
            print("EPOCH {}:".format(epoch_number + 1))

            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            avg_loss = self.train_one_epoch(epoch_number)

            running_vloss = 0.0
            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            self.model.eval()

            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                for i, vdata in enumerate(self.validation_loader):
                    vinputs, vlabels = vdata
                    vinputs = vinputs.to(self.device)
                    vlabels = vlabels.to(self.device)
                    voutputs = self.model(vinputs).reshape(-1)
                    vloss = self.model.loss_fn(voutputs, vlabels)
                    running_vloss += vloss

                    avg_vloss = running_vloss / (i + 1)
                print(f"LOSS\n\ttrain {avg_loss}\n\tvalid {avg_vloss}\n\n")

            # Log the running loss averaged per batch
            # for both training and validation
            print(
                "Training vs. Validation Loss",
                {"Training": avg_loss, "Validation": avg_vloss},
                epoch_number + 1,
            )

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = "model_{}_{}".format(timestamp, epoch_number)
                torch.save(self.model.state_dict(), model_path)

            epoch_number += 1
