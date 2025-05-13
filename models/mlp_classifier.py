from typing import List
import torch


class ADClassifier(torch.nn.Module):
    def __init__(self, gene_input_dim: int, cell_type_input_dim: int, hidden_dims: List[int]):
        super(ADClassifier, self).__init__()

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
