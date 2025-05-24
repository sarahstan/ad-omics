from typing import Tuple
import lightning as ltn
import torch
import numpy as np
from models.torch.transformer_classifier import ADPredictionModel


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
        self.l1_lambda = l1_lambda
        self.loss_fn = torch.nn.BCELoss()

        self.model = ADPredictionModel(
            embed_dim: hidden_dims
            num_heads: int,
            ff_dim: int,
            num_layers: int,
            max_seq_len: int,
            dropout: float = 0.1
        )

    def get_l1_loss(self) -> torch.Tensor:
        """Calculate the L1 loss for all parameters in the model."""
        l1_loss = 0.0
        for param in self.model.parameters():
            l1_loss += torch.sum(torch.abs(param))
        return self.l1_lambda * l1_loss

    def _get_loss(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        prefix: str = "",
        write_to_log: bool = True,
    ):
        x, y = batch
        outputs = self(x).reshape(-1)

        bce_loss = self.loss_fn(outputs, y)
        l1_loss = self.get_l1_loss()
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
        """Calculate percentage of zero weights at multiple thresholds"""
        total_params = 0
        # Check multiple thresholds
        thresholds = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
        counts = {t: 0 for t in thresholds}

        for name, param in self.model.named_parameters():
            if "weight" in name:
                total_params += param.numel()
                for t in thresholds:
                    counts[t] += (param.abs() < t).sum().item()

        # Log each threshold but maintain original behavior
        original_sparsity = 100.0 * counts[1e-6] / total_params if total_params > 0 else 0.0

        # Log additional metrics
        for t in thresholds:
            sparsity = 100.0 * counts[t] / total_params if total_params > 0 else 0.0
            self.log(f"sparsity_1e{int(np.log10(t))}", sparsity)

        return original_sparsity  # Return the original metric for consistency

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
