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
        input_dim: int | None,
        hidden_dim: int,
        activation: str,
        hidden_dim_2: int | None = None,
        use_batch_norm: bool = False,
        dropout: float = 0.0,
        numerical_dim: int | None = None,
        categorical_cardinalities: list[int] | None = None,
        embedding_dims: list[int] | None = None

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
        
        if embedding_dims is None:
            raise ValueError(
                "embedding_dims are required."
            )

        self.uses_embeddings = (
            categorical_cardinalities is not None
        )

        if not self.uses_embeddings:
            if input_dim is None or input_dim < 1:
                raise ValueError(
                    "input_dim must be positive "
                    "when embeddings are disabled."
                )

            self.embeddings = None
            network_input_dim = input_dim

        else:
            if numerical_dim is None or numerical_dim < 1:
                raise ValueError(
                    "numerical_dim must be positive "
                    "when embeddings are enabled."
                )

            if (
                len(categorical_cardinalities)
                != len(embedding_dims)
            ):
                raise ValueError(
                    "Each categorical feature must "
                    "have one embedding dimension."
                )

            self.embeddings = nn.ModuleList(
                [
                    nn.Embedding(
                        num_embeddings=cardinality,
                        embedding_dim=embedding_dim,
                        padding_idx=0
                    )
                    for cardinality, embedding_dim in zip(
                        categorical_cardinalities,
                        embedding_dims
                    )
                ]
            )

            network_input_dim = (
                numerical_dim + sum(embedding_dims)
            )

        layers = [
            nn.Linear(
                network_input_dim,
                hidden_dim
            )
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

    def forward(
        self,
        features: torch.Tensor,
        categorical_features: torch.Tensor | None = None,
    ) -> torch.Tensor:

        if not self.uses_embeddings:
            if categorical_features is not None:
                raise ValueError(
                    "Categorical features were provided "
                    "while embeddings are disabled."
                )
            combined_features = features

        else:
            if categorical_features is None:
                raise ValueError(
                    "Categorical features are required "
                    "when embeddings are enabled."
                )

            embedded_features = [
                embedding(
                    categorical_features[:, column_index]
                )
                for column_index, embedding in enumerate(self.embeddings)
            ]

            combined_features = torch.cat(
                [
                    features,
                    *embedded_features,
                ],
                dim=1,
            )

        predictions = self.network(combined_features)

        # (batch_size, 1) → (batch_size,)
        # target для регрессии имеет форму:
        # y.shape
        # [batch_size]
        return predictions.squeeze(-1)