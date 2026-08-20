import json
import math
import time
from pathlib import Path

import pandas as pd
from DL.optimizers import build_optimizer

from DL.data import (
    build_train_loader,
    prepare_fold,
)
from DL.dl_models.mlp import HousePriceMLP
from DL.trainer import (
    evaluate_log_rmse,
    fit_model,
)
from DL.tuning import (
    run_optuna_study,
    seed_everything,
)

from DL.schedulers import build_scheduler

from utils.experiment_artifacts import (
    save_split_indices,
)
from utils.experiment_logging import (
    save_cv_results,
    save_metrics,
)
from utils.validation import build_cv_splits

def run_nested_cv(
    *,
    X,
    y,
    config,
    device,
    experiment_dir: Path,
    logger,
) -> pd.DataFrame:
    """Выполнить nested cross-validation для DL-модели.

    На каждом внешнем фолде внутренний CV с Optuna выбирает
    гиперпараметры и рекомендуемое число эпох. Затем новая модель
    обучается на всей train-части внешнего фолда и оценивается на
    его validation-части.

    Returns:
        Таблица с RMSLE, параметрами и временем обучения каждого
        внешнего фолда.
    """

    outer_splits = build_cv_splits(
        X=X,
        y=y,
        cfg_validation=config.validation,
    )

    outer_cv_dir = experiment_dir / "outer_cv"

    save_split_indices(
        cv_splits=outer_splits,
        output_path=(outer_cv_dir / "split_indices.npz"),
    )

    outer_results = []
    base_seed = int(config.general.seed)

    for outer_index, (
        outer_train_indices,outer_valid_indices
        ) in enumerate(outer_splits):
        outer_fold = outer_index + 1

        logger.info(
            "Outer fold %d/%d started",
            outer_fold,
            len(outer_splits),
        )

        X_outer_train = X.iloc[outer_train_indices].reset_index(drop=True)
        y_outer_train = y.iloc[outer_train_indices].reset_index(drop=True)

        inner_splits = build_cv_splits(
            X=X_outer_train,
            y=y_outer_train,
            cfg_validation=config.optuna.inner_validation
        )

        outer_fold_dir = outer_cv_dir / f"fold_{outer_fold}"
        inner_study_dir = outer_fold_dir / "optuna"

        save_split_indices(
            cv_splits=inner_splits,
            output_path=inner_study_dir / "split_indices.npz"
        )

        study = run_optuna_study(
            X=X_outer_train,
            y=y_outer_train,
            cv_splits=inner_splits,
            config=config,
            device=device,
            study_dir=inner_study_dir,
            study_name=f"{config.model.name}_outer_fold_{outer_fold}"
        )

        best_params = dict(study.best_params)

        optimizer_name = str(best_params["optimizer"])

        learning_rate = float(best_params[f"{optimizer_name}_learning_rate"])

        recommended_epochs = study.best_trial.user_attrs.get("recommended_epochs")

        if recommended_epochs is None:
            raise ValueError(
                f"Outer fold {outer_fold}: best trial does not contain "
                "recommended_epochs."
            )

        recommended_epochs = int(recommended_epochs)

        if recommended_epochs < 1:
            raise ValueError(
                f"Outer fold {outer_fold}: "
                "recommended_epochs must be positive."
            )

        prepared_outer = prepare_fold(
            X=X,
            y=y,
            train_indices=outer_train_indices,
            valid_indices=outer_valid_indices,
            config=config,
        )

        use_embeddings = (
            prepared_outer["categorical_cardinalities"] is not None
        )


        outer_seed = base_seed + outer_index

        seed_everything(outer_seed)

        train_loader, generator = (
            build_train_loader(
                X=prepared_outer["X_train"],
                X_cat=prepared_outer["X_cat_train"],
                y=prepared_outer["y_train"],
                batch_size=int(best_params["batch_size"]),
                shuffle=True,
                seed=outer_seed,
                num_workers=int(config.dl.training.num_workers),
                pin_memory=bool(config.dl.training.pin_memory),
                drop_last=bool(config.dl.training.drop_last)
            )
        )

        valid_loader, _ = build_train_loader(
            X=prepared_outer["X_valid"],
            X_cat=prepared_outer["X_cat_valid"],
            y=prepared_outer["y_valid"],
            batch_size=int(best_params["batch_size"]),
            shuffle=False,
            seed=outer_seed,
            num_workers=int(config.dl.training.num_workers),
            pin_memory=bool(config.dl.training.pin_memory),
            drop_last=False,
        )

        model_params = {
            "hidden_dim": int(best_params["hidden_dim"]),
            "hidden_dim_2": int(best_params["hidden_dim_2"]),
            "activation": str(best_params["activation"]),
            "use_batch_norm": bool(config.dl.model.batch_norm.enabled),
            "dropout": float(best_params["dropout"])
        }

        # Для OHE
        if not use_embeddings:
            model_params.update(
                {
                    "input_dim": int(
                        prepared_outer["X_train"].shape[1]
                        ),
                }
            )
        # Для embeddings
        else:
            cardinalities = (prepared_outer["categorical_cardinalities"])

            model_params.update(
                {
                    "input_dim": None,

                    "numerical_dim": int(
                        prepared_outer[
                            "X_train"
                        ].shape[1]
                    ),

                    "categorical_cardinalities": list(
                        cardinalities
                    ),

                    "embedding_dims": [
                        int(best_params["embedding_dim"])
                        for _ in cardinalities
                    ],
                }
            )

        model = HousePriceMLP(**model_params).to(device)

        optimizer_params = {
            "name": optimizer_name,
            "lr": learning_rate,
            "weight_decay": float(best_params["weight_decay"]),
        }

        optimizer = build_optimizer(
            parameters=model.parameters(),
            **optimizer_params,
        )

        scheduler = build_scheduler(
            optimizer=optimizer,
            cfg_scheduler=config.dl.scheduler,
        )

        start_time = time.perf_counter()

        fit_model(
            model=model,
            train_loader=train_loader,
            valid_loader=None,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            max_epochs=recommended_epochs,
            model_params=model_params,
            optimizer_params=optimizer_params,
            gradient_clip_norm=float(config.dl.training.gradient_clip_norm),
            trial=None,
            checkpoint_dir=None,
            dataloader_generator=generator,
        )

        outer_score = evaluate_log_rmse(
            model=model,
            loader=valid_loader,
            device=device,
        )

        fit_time = (time.perf_counter() - start_time)

        if not math.isfinite(outer_score):
            raise FloatingPointError(
                f"Outer fold {outer_fold} "
                f"produced score={outer_score}."
            )

        outer_results.append(
            {
                "fold": outer_fold,
                "validation_score": float(
                    outer_score
                ),
                "recommended_epochs": (
                    recommended_epochs
                ),
                "best_params": json.dumps(
                    best_params,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "fit_time_seconds": float(
                    fit_time
                ),
            }
        )

        logger.info(
            "Outer fold %d RMSLE: %.5f",
            outer_fold,
            outer_score,
        )

    fold_results = pd.DataFrame(outer_results)
    mean_score = (fold_results["validation_score"].mean())
    std_score = (fold_results["validation_score"].std())

    cv_results_path = save_cv_results(
        fold_results=fold_results,
        experiment_dir=experiment_dir,
    )

    metrics_path = save_metrics(
        metric_name=str(config.metric.name),
        mean_score=mean_score,
        std_score=std_score,
        experiment_dir=experiment_dir,
    )

    logger.info(
        "Nested outer CV %s: %.5f +/- %.5f",
        config.metric.name,
        mean_score,
        std_score,
    )

    logger.info(
        "Outer CV results saved to: %s",
        cv_results_path,
    )

    logger.info(
        "Nested CV metrics saved to: %s",
        metrics_path,
    )
    
    return fold_results