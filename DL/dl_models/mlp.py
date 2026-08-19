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
        activation: str,
        hidden_dim_2: int | None = None,
        use_batch_norm: bool = False,
        dropout: float = 0.0
    ):

        super().__init__() 

        if input_dim < 1:
            raise ValueError("input_dim must be positive.")
        
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive.")
        
        if hidden_dim_2 is not None and hidden_dim_2 < 1:
            raise ValueError(
                "hidden_dim_2 must be positive."
            )
        
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the interval [0.0, 1.0).")

        layers = [
            nn.Linear(input_dim, hidden_dim)
        ]

        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        
        layers.append(get_activation(activation))
        layers.append(nn.Dropout(p=dropout))

        last_hidden_dim = hidden_dim

        if hidden_dim_2 is not None:
            layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim_2),
                    get_activation(activation),
                    nn.Dropout(p=dropout)
                ]
            )

            last_hidden_dim = hidden_dim_2

        layers.append(nn.Linear(last_hidden_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        predictions = self.network(features)

        # (batch_size, 1) → (batch_size,)
        # target для регрессии имеет форму:
        # y.shape
        # [batch_size]
        return predictions.squeeze(-1)