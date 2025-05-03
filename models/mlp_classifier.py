from typing import List
import torch


class ADClassifier(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int]):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        layers: List = []
        current_input_dim = input_dim
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


# class ADClassifier(torch.nn.Module):
#     def __init__(self, input_dim: int, hidden_dims: list[int]):
#         super(ADClassifier, self).__init__()
#         self.input_dim = input_dim
#         self.hidden_dims = hidden_dims

#         # Create a list of layers
#         layers = []
#         in_dim = input_dim
#         for dim in hidden_dims:
#             layers.append(torch.nn.Linear(in_dim, dim))
#             layers.append(torch.nn.ReLU())
#             in_dim = dim

#         # Add the final output layer
#         layers.append(torch.nn.Linear(in_dim, 1))
#         layers.append(torch.nn.Sigmoid())

#         # Combine all layers into a sequential model
#         self.model = torch.nn.Sequential(*layers)
#         self.loss_fn = torch.nn.BCELoss()
