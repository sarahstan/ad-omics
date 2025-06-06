from typing import Tuple
import lightning as ltn
import torch
import numpy as np
from models.transformer.classifier import ADPredictionModel
from configs import (
    CellStateEncoderConfig,
    ScRNATransformerConfig,
    TrainingConfig,
)


class ADClassifierLightning(ltn.LightningModule):
    def __init__(
        self,
        cell_state_encoder_config: CellStateEncoderConfig,
        scrna_transformer_config: ScRNATransformerConfig,
        training_config: TrainingConfig,
    ):
        super().__init__()
        self.training_config = training_config
        self.cell_state_encoder_config = cell_state_encoder_config
        self.scrna_transformer_config = scrna_transformer_config
        self.learning_rate = self.training_config.learning_rate
        self.l1_lambda = self.training_config.l1_lambda

        self.save_hyperparameters()

        self.loss_fn = torch.nn.BCEWithLogitsLoss()  # Changed to use logits version

        self.model = ADPredictionModel(
            cell_state_encoder_config=cell_state_encoder_config,
            scrna_transformer_config=scrna_transformer_config,
        )

    def get_l1_loss(self) -> torch.Tensor:
        """Calculate the L1 loss for all parameters in the model."""
        l1_loss = 0.0
        for param in self.model.parameters():
            l1_loss += torch.sum(torch.abs(param))
        return self.l1_lambda * l1_loss

    def _get_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        prefix: str = "",
        write_to_log: bool = True,
    ):
        bce_loss = self.loss_fn(logits, labels)
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
        gene_indices, gene_counts, cell_type, labels, attention_mask = batch
        logits, _ = self.model(
            gene_indices=gene_indices,
            gene_values=gene_counts,
            cell_type_indices=cell_type,
            attention_mask=attention_mask,
        )
        return self._get_loss(logits, labels, prefix="train")

    def validation_step(self, batch, batch_idx):
        gene_indices, gene_counts, cell_type, labels, attention_mask = batch
        logits, _ = self.model(
            gene_indices=gene_indices,
            gene_values=gene_counts,
            cell_type_indices=cell_type,
            attention_mask=attention_mask,
        )
        return self._get_loss(logits, labels, prefix="val")

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

    def forward(self, gene_indices, gene_values, cell_type_indices, attention_mask=None):
        logits, attention_weights = self.model(
            gene_indices, gene_values, cell_type_indices, attention_mask
        )
        return logits
