from pathlib import Path

import joblib
from omegaconf import DictConfig, OmegaConf
from sklearn.base import BaseEstimator

def prepare_experiment_dir(config) -> Path:
    experiment_dir = (
        Path(config.paths.trained_models)
        / str(config.model.name)
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