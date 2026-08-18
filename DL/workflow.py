from pathlib import Path

import joblib
import torch

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
    to_float32_array,
    transform_target,
)

from utils.validation import build_cv_splits
from utils.experiment_artifacts import save_split_indices

from DL.nested_cv import run_nested_cv
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

    run_nested_cv(
        X=X_train,
        y=y_train,
        config=config,
        device=device,
        experiment_dir=experiment_dir,
        logger=logger,
    )

    final_inner_splits = build_cv_splits(
        X=X_train,
        y=y_train,
        cfg_validation=config.optuna.inner_validation
    )

    final_optuna_dir = (
        experiment_dir / "final_optuna"
    )

    # Сохранение индексов всех фолдов.
    # Сборка словаря.
    split_arrays = {}

    save_split_indices(
        cv_splits=final_inner_splits,
        output_path=final_optuna_dir / "split_indices.npz"
    )


    # Optuna
    logger.info("Final Optuna study started")

    final_study = run_optuna_study(
        X=X_train,
        y=y_train,
        cv_splits=final_inner_splits,
        config=config,
        device=device,
        study_dir=final_optuna_dir,
        study_name=f"{config.model.name}_final"
    )

    best_params = dict(final_study.best_params)

    recommended_epochs = final_study.best_trial.user_attrs.get("recommended_epochs")

    if recommended_epochs is None:
        raise ValueError(
            "Final trial does not contain recommended_epochs."
        )

    final_epochs = int(recommended_epochs)

    if final_epochs < 1:
        raise ValueError(
            "recommended_epochs must be positive."
        )

    # preprocessing на всём train.
    final_feature_pipeline = build_dl_feature_pipeline(config)

    X_full = final_feature_pipeline.fit_transform(X_train)
    X_full = to_float32_array(X_full)

    y_full = transform_target(y_train)

    # Сохранение.
    feature_pipeline_path = experiment_dir / "feature_pipeline.joblib"

    joblib.dump(
        final_feature_pipeline,
        feature_pipeline_path,
        compress=3
    )

    # final DataLoader.
    final_loader, generator = (
        build_train_loader(
            X=X_full,
            y=y_full,
            batch_size=int(best_params["batch_size"]),
            shuffle=True,
            seed=seed,
            num_workers=int(config.dl.training.num_workers),
            pin_memory=bool(config.dl.training.pin_memory),
            drop_last=bool(config.dl.training.drop_last)
        )
    )

    # Новая финальная модель.
    seed_everything(seed)
    
    model_params = {
        "input_dim": int(X_full.shape[1]),
        "hidden_dim": int(best_params["hidden_dim"]),
        "activation": str(best_params["activation"])
    }
    
    final_model = HousePriceMLP(
        **model_params
    ).to(device)

    optimizer_params = {
        "lr": float(best_params["learning_rate"]),
        "weight_decay": float(best_params["weight_decay"])
    }

    optimizer = AdamW(
        final_model.parameters(),
        **optimizer_params
    )

    # final training.
    checkpoints_dir = experiment_dir / "checkpoints"

    result = fit_model(
        model=final_model,
        train_loader=final_loader,
        valid_loader=None,
        optimizer=optimizer,
        device=device,
        max_epochs=final_epochs,
        model_params=model_params,
        optimizer_params=optimizer_params,
        gradient_clip_norm=float(config.dl.training.gradient_clip_norm),
        checkpoint_dir=checkpoints_dir,
        dataloader_generator=generator
    )

# Поскольку финальный fit использует весь train, best.pt здесь не появится — validation нет.
# Будет только last.pt , который лучше пересохранить с названием final.pt.
    final_checkpoint = build_checkpoint(
        model=final_model,
        optimizer=optimizer,
        epoch=result["last_epoch"],
        global_step=result["global_step"],
        model_params=model_params,
        optimizer_params=optimizer_params,
        best_valid_rmsle=None,
        best_epoch=None,
        history=result["history"],
        dataloader_generator=generator
    )

    final_checkpoint_path = save_checkpoint(
        final_checkpoint, checkpoints_dir / "final.pt"
    )

    # Inference.
    if config.inference.enabled:
        submission_path = create_dl_submission(
            checkpoint_path=final_checkpoint_path,
            feature_pipeline_path=feature_pipeline_path,
            X_test=X_test,
            test_ids=test_ids,
            id_column=str(config.id_column),
            prediction_column=str(config.inference.prediction_column),
            output_path=(
                experiment_dir / str(config.inference.submission_filename)
            ),
            device=device,
            batch_size=int(best_params["batch_size"]),
        )

        logger.info("Submission saved to: %s", submission_path)

