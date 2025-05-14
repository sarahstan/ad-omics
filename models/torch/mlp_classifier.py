from typing import List
import torch


class ADClassifier(torch.nn.Module):
    def __init__(
        self,
        gene_input_dim: int,
        cell_type_input_dim: int,
        hidden_dims: List[int],
        l1_lambda: float = 0.0001,
    ):
        """
        A simple MLP classifier for Alzheimer's disease classification.

        Args:
            gene_input_dim (int): The number of input features for gene expression data.
            cell_type_input_dim (int): The number of input features for cell type data.
            hidden_dims (List[int]): List of integers representing number of neurons in each layer.
            l1_lambda (float): The L1 regularization strength. Default is 0.0001.
        """
        super(ADClassifier, self).__init__()

        self.l1_lambda = l1_lambda

        layers: List = []
        current_input_dim = gene_input_dim + cell_type_input_dim
        for layer_dim in hidden_dims:
            layers.append(torch.nn.Linear(current_input_dim, layer_dim))
            layers.append(torch.nn.ReLU())
            current_input_dim = layer_dim

        layers.append(torch.nn.Linear(layer_dim, 1))
        layers.append(torch.nn.Sigmoid())

        self.model = torch.nn.Sequential(*layers)
        self.loss_fn = torch.nn.BCELoss()

    def get_l1_loss(self) -> torch.Tensor:
        """
        Calculate the L1 loss for all parameters in the model.

        Returns:
            torch.Tensor: The L1 loss.
        """
        l1_loss = 0.0
        for param in self.parameters():
            l1_loss += torch.sum(torch.abs(param))
        return self.l1_lambda * l1_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
