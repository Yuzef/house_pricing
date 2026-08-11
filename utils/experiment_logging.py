import json
import logging
from pathlib import Path

import pandas as pd

def setup_experiment_logger(experiment_dir: Path) -> logging.Logger:
    # Даём название
    logger = logging.getLogger("house_pricing.experiment")
    logger.setLevel(logging.INFO)
    # не передавай эти сообщения дальше родительскому/root logger.
    logger.propagate = False 

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt=("%(asctime)s | %(levelname)s | %(message)s"),
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(
        experiment_dir / "experiment.log",
        mode="w",
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def save_cv_results(
    fold_results: pd.DataFrame,
    experiment_dir: Path
) -> Path:
    results_path = experiment_dir / "cv_results.csv"

    fold_results.to_csv(results_path, index=False)

    return results_path

def save_metrics(
    metric_name: str,
    mean_score: float,
    std_score: float,
    experiment_dir: Path
) -> Path:
    metrics_path = experiment_dir / "metrics.json"

    metrics = {
        "metric": metric_name,
        "cv_mean": float(mean_score),
        "cv_std": float(std_score)
    }

    with metrics_path.open(mode="w", encoding="utf-8") as file:
        json.dump(
            metrics,
            file,
            indent=2,
            ensure_ascii=False
        )
    
    return metrics_path

def save_grid_search_results(grid_search, experiment_dir) -> Path:
    results_path = experiment_dir / "grid_search_results.csv"

    pd.DataFrame(grid_search.cv_results_).to_csv(
        results_path,
        index=False
        )
    
    return results_path

def save_best_params(best_params, experiment_dir) -> Path:
    best_params_path = experiment_dir / "best_params.json"

    with best_params_path.open(
        mode="w",
        encoding="utf-8"
    ) as file:
        json.dump(
            best_params,
            file,
            indent=2,
            ensure_ascii=False
        )
    
    return best_params_path
