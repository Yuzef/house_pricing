import json
import random
import math
from pathlib import Path

import numpy as np
import optuna
import torch
from optuna.trial import TrialState
from DL.optimizers import build_optimizer
from DL.schedulers import build_scheduler

from DL.data import (
    build_train_loader,
    prepare_fold,
)
from DL.dl_models.mlp import HousePriceMLP
from DL.trainer import fit_model

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_objective(
    *,
    X,
    y,
    cv_splits,
    config,
    device,
):
    def objective(trial):

        base_seed = int(config.general.seed)

        # Гиперпараметры выбираются один раз на trial.
        hidden_dim = trial.suggest_categorical(
            "hidden_dim",
            list(config.optuna.search_space.hidden_dim)
        )

        hidden_dim_2 = trial.suggest_categorical(
            "hidden_dim_2",
            list(config.optuna.search_space.hidden_dim_2),
        )

        activation = trial.suggest_categorical(
            "activation",
            list(config.optuna.search_space.activation),
        )

        dropout = trial.suggest_categorical(
            "dropout",
            list(config.optuna.search_space.dropout),
        )

        batch_size = trial.suggest_categorical(
            "batch_size",
            list(config.optuna.search_space.batch_size)
        )

        optimizer_name = trial.suggest_categorical(
            "optimizer",
            list(config.optuna.search_space.optimizer),
        )

        learning_rate_cfg = (
            config.optuna.search_space.learning_rate[optimizer_name]
        )

        learning_rate = trial.suggest_float(
            f"{optimizer_name}_learning_rate",
            low=float(learning_rate_cfg.low),
            high=float(learning_rate_cfg.high),
            log=bool(learning_rate_cfg.log),
        )

        weight_decay = trial.suggest_float(
            "weight_decay",
            low=float(config.optuna.search_space.weight_decay.low),
            high=float(config.optuna.search_space.weight_decay.high),
            log=bool(config.optuna.search_space.weight_decay.log)
        )

        fold_scores = []
        fold_epochs = []

        for fold_index, (train_indices, valid_indices) in enumerate(cv_splits):
            fold_seed = base_seed + fold_index
            seed_everything(fold_seed)

            prepared_data = prepare_fold(
                X=X,
                y=y,
                train_indices=train_indices,
                valid_indices=valid_indices,
                config=config,
            )
    
            train_loader, generator = build_train_loader(
                X=prepared_data["X_train"],
                y=prepared_data["y_train"],
                batch_size=batch_size,
                shuffle=True,
                seed=fold_seed,
                num_workers=int(config.dl.training.num_workers),
                pin_memory=bool(config.dl.training.pin_memory),
                drop_last=bool(config.dl.training.drop_last)
            )

            valid_loader, _ = build_train_loader(
                X=prepared_data["X_valid"],
                y=prepared_data["y_valid"],
                batch_size=batch_size,
                shuffle=False,
                seed=fold_seed,
                num_workers=int(config.dl.training.num_workers),
                pin_memory=bool(config.dl.training.pin_memory),
                drop_last = False
            )

            model_params = {
                "input_dim": int(prepared_data["X_train"].shape[1]),
                "hidden_dim": int(hidden_dim),
                "hidden_dim_2": int(hidden_dim_2),
                "activation": str(activation),
                "use_batch_norm": bool(config.model.params.batch_norm.enabled),
                "dropout": float(dropout)
            }

            model = HousePriceMLP(**model_params).to(device)

            optimizer_params = {
                "name": str(optimizer_name),
                "lr": float(learning_rate),
                "weight_decay": float(weight_decay),
            }

            optimizer = build_optimizer(
                parameters=model.parameters(),
                **optimizer_params
            )

            scheduler = build_scheduler(
                optimizer=optimizer,
                cfg_scheduler=config.dl.scheduler
            )

            # Запуск Trainer:
            result = fit_model(
                model=model,
                train_loader=train_loader,
                valid_loader=valid_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                max_epochs=int(config.dl.training.max_epochs),
                model_params=model_params,
                optimizer_params=optimizer_params,
                gradient_clip_norm=float(
                    config.dl.training.gradient_clip_norm
                ),
                trial=None,
                checkpoint_dir=None,
                dataloader_generator=generator
            )

            score = result["best_valid_rmsle"]
            best_epoch = result["best_epoch"]

            if (
                score is None
                or not math.isfinite(score)
                or best_epoch is None
            ):
                raise FloatingPointError(
                    f"Fold {fold_index} produced an invalid result: "
                    f"score={score}, best_epoch={best_epoch}."
                )
            
            fold_scores.append(float(score))
            # best_epoch имеет нумерацию с нуля.
            fold_epochs.append(int(best_epoch) + 1)

            # Fold-level pruning
            running_mean = float(np.mean(fold_scores))

            trial.set_user_attr("fold_scores", list(fold_scores))
            trial.set_user_attr("fold_epochs", list(fold_epochs))

            trial.report(running_mean, step=fold_index)

            if trial.should_prune():
                raise optuna.TrialPruned()
        
        mean_score = float(np.mean(fold_scores))
        recommended_epochs = int(round(np.median(fold_epochs)))

        trial.set_user_attr("fold_scores", list(fold_scores))
        trial.set_user_attr("fold_epochs", list(fold_epochs))
        trial.set_user_attr("recommended_epochs", recommended_epochs)

        return mean_score
    
    return objective

def run_optuna_study(
    *,
    X,
    y,
    cv_splits,
    config,
    device,
    study_dir: Path,
    study_name: str
):

    study_dir.mkdir(parents=True, exist_ok=True)

    database_path = (study_dir / "study.db").resolve()

    storage_url = f"sqlite:///{database_path}"

    sampler = optuna.samplers.TPESampler(
        seed=int(config.optuna.sampler.seed),
        n_startup_trials=int(config.optuna.sampler.n_startup_trials)
    )

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=int(config.optuna.pruner.n_startup_trials),
        n_warmup_steps=int(config.optuna.pruner.n_warmup_steps)
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True
    )

    finished_trials = sum(
        trial.state in {TrialState.COMPLETE, TrialState.PRUNED}
        for trial in study.trials
    )

    remaining_trials = max(
        0, int(config.optuna.target_n_trials) - finished_trials
    )

    if remaining_trials > 0:
        study.optimize(
            func=make_objective(
                X=X,
                y=y,
                cv_splits=cv_splits,
                config=config,
                device=device,
            ),
            n_trials=remaining_trials,
            timeout=(
                None
                if config.optuna.timeout_seconds is None
                else int(config.optuna.timeout_seconds)
            ),
            n_jobs=int(config.optuna.n_jobs),
            gc_after_trial=True
        )

    study.trials_dataframe().to_csv(
        study_dir / "trials.csv",
        index=False
    )

    complete_trials = study.get_trials(
        deepcopy=False,
        states=(TrialState.COMPLETE,),
    )

    if not complete_trials:
        raise RuntimeError(
            f"Optuna study {study_name} has no completed trials."
        )

    best_result = {
        "trial_number": study.best_trial.number,
        "value": study.best_value,
        "params": study.best_params,
        "fold_scores": study.best_trial.user_attrs.get("fold_scores"),
        "fold_epochs": study.best_trial.user_attrs.get("fold_epochs"),
        "recommended_epochs": study.best_trial.user_attrs.get("recommended_epochs")
    }

    with (study_dir / "best_params.json").open("w", encoding="utf-8") as file:
        json.dump(
            best_result,
            file,
            indent=2,
            ensure_ascii=False
        )
    
    return study