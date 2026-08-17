# Linear(input_dim → hidden_dim)
# ReLU
# Linear(hidden_dim → 1)

import torch
from torch import nn

def get_activation(name: str) -> nn.Module:
    activation_classes = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU
    }

    try:
        activation_class = activation_classes[name]
    except KeyError as error:
        raise ValueError(
            f"Unknown activation: {name}."
            f"Available: {tuple(activation_classes)}"
        ) from error

    return activation_class()

class HousePriceMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        activation: str
    ):
        super().__init__() 

        if input_dim < 1:
            raise ValueError("input_dim must be positive.")
        
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive.")

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            get_activation(activation),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        predictions = self.network(features)

        # (batch_size, 1) → (batch_size,)
        # target для регрессии имеет форму:
        # y.shape
        # [batch_size]
        return predictions.squeeze(-1)