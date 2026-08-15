# Сохраняет состояния компонентов.

from pathlib import Path

import torch
from torch.cpu import is_available
from torch.utils.data import dataloader

def build_checkpoint(
    *, # все аргументы после * разрешено передавать только по имени.
    model,
    optimizer,
    epoch: int,
    global_step: int,
    model_params: dict,
    optimizer_params: dict,
    best_valid_rmsle,
    best_epoch,
    history: list,
    dataloader_generator=None,
    scheduler=None,
    scaler=None
) -> dict:
    return {
        "checkpoint_version": 1,
        "epoch": int(epoch),
        "global_step": int(global_step),

        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),

        "scheduler_state_dict": (
            scheduler.state_dict()
            if scheduler is not None
            else None
        ),

        "scaler_state_dict": (
            scaler.state_dict()
            if scaler is not None
            else None
        ),

        "model_params": dict(model_params),
        "optimizer_params": dict(optimizer_params),

        "best_valid_rmsle": best_valid_rmsle,
        "best_epoch": best_epoch,
        "history": list(history),

        "torch_rng_state": torch.get_rng_state(),

        "cuda_rng_state": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),

        "dataloader_generator_state": (
            dataloader_generator.get_state()
            if dataloader_generator is not None
            else None
        ),

        "target_transform": "log1p"
    }

def save_checkpoint(checkpoint: dict, path: Path) -> Path:
    """
    Если процесс упадёт во время torch.save,
    предыдущий рабочий .pt не будет испорчен.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    
    temporary_path = path.with_suffix(path.with_suffix + ".tmp")

    torch.save(checkpoint, temporary_path)

    temporary_path.replace(path)

    return path

def read_checkpoint(path: Path, device) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}.")
    
    return torch.load(path, map_location=device, weights_only=True)

def restore_training_state(
    checkpoint: dict,
    model,
    optimizer,
    scheduler=None,
    scaler=None,
    dataloader_generator=None
) -> None:
    model.load_state_dict(checkpoint["model_state_dict"])
    
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if (
        scheduler is not None
        and ["scheduler_state_dict"] is not None
    ):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if (
        scaler is not None
        and checkpoint["scheduler_state_dict"] is not None
    ):
        scaler.load_state_dict(checkpoint["scheduler_state_dict"])

    torch.set_rng_state(checkpoint["torch_rng_state"])

    if (
        torch.cuda.is_available()
        and checkpoint["cuda_rng_state"] is not None
    ):
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])

    if (
        dataloader_generator is not None
        and checkpoint["dataloader_generator_state"] is not None
    ):
        dataloader_generator.set_state(checkpoint["dataloader_generator_state"])