from pathlib import Path

import numpy as np
import joblib
from omegaconf import OmegaConf

def prepare_experiment_dir(config) -> Path:
    experiment_dir = (
        Path(config.paths.trained_models)
        / str(config.general.experiment_name)
    )

    experiment_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    OmegaConf.save(
        config=config,
        f=experiment_dir / "config.yaml",
        resolve=True
    )

    return experiment_dir

def save_model(model, experiment_dir: Path) -> Path:
    model_path = experiment_dir / "model.joblib"

    joblib.dump(
        model,
        model_path,
        compress=3
    )

    return model_path

def save_split_indices(
    *,
    cv_splits,
    output_path: Path,
) -> Path:
    """
    Сохранить train/validation-индексы всех CV-фолдов
    в NPZ-файл.
    """
    split_arrays = {}

    for fold_index, (train_indices, valid_indices) in enumerate(cv_splits):
        fold_number = fold_index + 1

        split_arrays[f"fold_{fold_number}_train_indices"] = train_indices
        split_arrays[f"fold_{fold_number}_valid_indices"] = valid_indices
    
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(output_path, **split_arrays)

    return output_path