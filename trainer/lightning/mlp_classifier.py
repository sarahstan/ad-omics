from typing import Tuple
import lightning as ltn
import torch
from models.torch.mlp_classifier import ADClassifier


class ADClassifierLightning(ltn.LightningModule):
    def __init__(
        self,
        gene_input_dim: int,
        cell_type_input_dim: int,
        hidden_dims: list[int],
        learning_rate: float = 0.001,
        l1_lambda: float = 0.0001,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        self.model = ADClassifier(
            gene_input_dim=gene_input_dim,
            cell_type_input_dim=cell_type_input_dim,
            hidden_dims=hidden_dims,
            l1_lambda=l1_lambda,
        )

    def forward(self, x):
        return self.model(x)

    def _get_loss(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        prefix: str = "",
        write_to_log: bool = True,
    ):
        x, y = batch
        outputs = self(x).reshape(-1)

        # Get base loss and L1 loss separately
        bce_loss = self.model.loss_fn(outputs, y)
        l1_loss = self.model.get_l1_loss()
        total_loss = bce_loss + l1_loss

        if write_to_log:
            prefix = prefix + "_" if prefix else ""
            self.log(f"{prefix}total_loss", total_loss)
            self.log(f"{prefix}bce_loss", bce_loss)
            self.log(f"{prefix}l1_loss", l1_loss)

            # Calculate and log sparsity percentage
            sparsity = self._calculate_sparsity()
            self.log(f"{prefix}weight_sparsity", sparsity)

        return total_loss

    def _calculate_sparsity(self):
        """Calculate percentage of zero weights in the model"""
        total_params = 0
        zero_params = 0

        for name, param in self.model.named_parameters():
            if "weight" in name:
                total_params += param.numel()
                zero_params += (param.abs() < 1e-6).sum().item()  # Count near-zero weights

        return 100.0 * zero_params / total_params if total_params > 0 else 0.0

    def training_step(self, batch, batch_idx):
        gene_expr, cell_type, labels = batch
        # Concatenate gene expression and cell type data
        x = torch.cat((gene_expr, cell_type), dim=1)
        return self._get_loss((x, labels), prefix="train")

    def validation_step(self, batch, batch_idx):
        gene_expr, cell_type, labels = batch
        # Concatenate gene expression and cell type data
        x = torch.cat((gene_expr, cell_type), dim=1)
        return self._get_loss((x, labels), prefix="val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def on_train_epoch_end(self):
        logger = self.logger.experiment  # TensorBoard SummaryWriter
        for name, param in self.model.named_parameters():
            # Log weights
            logger.add_histogram(f"weights/{name}", param, self.current_epoch)

            # Log gradients, if available
            if param.grad is not None:
                logger.add_histogram(f"gradients/{name}", param.grad, self.current_epoch)
