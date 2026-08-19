from torch.optim import (
    Adam,
    AdamW,
    Optimizer,
    RMSprop,
)


def build_optimizer(
    *,
    parameters,
    name: str,
    lr: float,
    weight_decay: float,
) -> Optimizer:

    optimizer_classes = {
        "adam": Adam,
        "adamw": AdamW,
        "rmsprop": RMSprop,
    }

    normalized_name = name.lower()

    try:
        optimizer_class = optimizer_classes[normalized_name]
    except KeyError as error:
        raise ValueError(
            f"Unknown optimizer: {name}. "
            f"Available: {tuple(optimizer_classes)}"
        ) from error

    return optimizer_class(
        parameters,
        lr=lr,
        weight_decay=weight_decay,
    )