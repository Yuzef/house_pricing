from pathlib import Path

import joblib
import numpy as np
import torch
from torch.cpu import is_available
from torch.optim import AdamW

from DL.checkpoints import (
    build_checkpoint,
    read_checkpoint,
    restore_training_state,
    save_checkpoint,
)
from DL.data import (
    build_dl_feature_pipeline,
    build_train_loader,
    prepare_tuning_split,
    to_float32_array,
    transform_target,
)
from DL.dl_models.mlp import HousePriceMLP
from DL.inference import create_dl_submission
from DL.trainer import fit_model
from DL.tuning import (
    run_optuna_study,
    seed_everything,
)

def resolve_device(device_name: str):
    if device_name != "auto":
        return torch.device(device_name)
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")

def run_dl_experiment(
    *,
    config,
    experiment_dir: Path,
    logger,
    X_train,
    y_train,
    X_test,
    test_ids
) -> None:
    seed = int(config.general.seed)
    seed_everything(seed)

    device = resolve_device(str(config.dl.training.device))

    logger.info("PyTorch device: %s", device)

    # Tuning data
    prepared_data = prepare_tuning_split(
        X=X_train,
        y=y_train,
        config=config
    )

    np.savez_compressed(
        experiment_dir / "split_indices.npz",
        train_indices=prepared_data["train_indices"],
        valid_indices=prepared_data["valid_indices"]
    )

    # Optuna
    study = run_optuna_study(
        prepared_data=prepared_data,
        config=config,
        device=device,
        experiment_dir=experiment_dir
    )

    best_params = dict(study.best_params)

    best_epoch = study.best_trial.user_attrs.get("best_epoch")

    if best_epoch is None:
        raise ValueError(
            "Best trial does not contain best_epoch."
        )

    final_epochs = int(best_epoch) + 1

    


