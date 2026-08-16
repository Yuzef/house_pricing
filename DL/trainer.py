import math
import optuna
from operator import is_
from pathlib import Path

from sklearn.metrics import mean_squared_error
import torch
from torch import nn

from DL.checkpoints import (
    build_checkpoint,
    save_checkpoint,
)

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion, # ф-ция ошибки в виде объекта nn
    device,
    gradient_clip_norm
):
    model.train()

    total_loss = 0.0
    total_objects = 0
    global_step_increment = 0

    for features, targets in loader:
        features = features.to(device, non_blocking=True)

        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        predictions = model(features)
        loss = criterion(predictions, targets)

        loss.backward()

        if gradient_clip_norm is not None:
            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=gradient_clip_norm
            )

        optimizer.step()

        batch_size = features.shape[0]

        total_loss += loss.item() * batch_size
        total_objects += batch_size
        global_step_increment += 1

    return total_loss / total_objects, global_step_increment

# Validation
def evaluate_log_rmse(model, loader, device) -> float:
    """
    Считает RMSE в логарифмическом пространстве.
    """
    model.eval()

    squared_error_sum = 0.0
    total_objects = 0

    # Отключение градиентов.
    with torch.inference_mode():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)

            predictions = model(features)

            # Все predictions меньше 0 заменить на 0, т.к. SalePrice >= 0
            predictions = predictions.clamp_min(0.0)

            squared_error_sum += torch.sum((predictions - targets) ** 2).item()

            total_objects += features.shape[0]

    mean_squared_error = squared_error_sum / total_objects

    return math.sqrt(mean_squared_error)

def fit_model(
    *,
    model,
    train_loader,
    optimizer,
    device,
    max_epochs: int,
    model_params: dict,
    optimizer_params: dict,
    gradient_clip_norm,
    valid_loader=None,
    trial=None,
    checkpoint_dir: Path | None = None,
    dataloader_generator=None,
    start_epoch: int = 0,
    global_step: int = 0,
    initial_best_score: float = float("inf"),
    initial_best_epoch=None,
    initial_history=None,
):
    criterion = nn.MSELoss()

    best_score = initial_best_score
    best_epoch = initial_best_epoch
    history = list(initial_history or [])

    for epoch in range(start_epoch, max_epochs):
        train_loss, step_increment = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            gradient_clip_norm=gradient_clip_norm
        )

        global_step += step_increment

        valid_rmsle = None

        if valid_loader is not None:
            valid_rmsle = evaluate_log_rmse(
                model=model,
                loader=valid_loader,
                device=device
            )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_rmsle": valid_rmsle
        }

        history.append(epoch_record)

        is_best = (
            valid_rmsle is not None
            and valid_rmsle < best_score
        )

        if is_best:
            best_score = valid_rmsle
            best_epoch = epoch

        checkpoint = build_checkpoint(
           model=model,
           optimizer=optimizer,
           epoch=epoch,
           global_step=global_step,
           model_params=model_params,
           optimizer_params=optimizer_params,
           best_valid_rmsle=(
            None
            if math.isinf(best_score)
            else best_score
           ),
           best_epoch=best_epoch,
           history=history,
           dataloader_generator=dataloader_generator
        )

        if checkpoint_dir is not None:
            save_checkpoint(checkpoint, checkpoint_dir / "last.pt")

            if is_best:
                save_checkpoint(checkpoint, checkpoint_dir / "best.pt")

        if trial is not None and valid_rmsle is not None:
            trial.report(valid_rmsle, step=epoch)

            if trial.should_prune():
                trial.set_user_attr("best_epoch", best_epoch)

                raise optuna.TrialPruned()
            
            
        
    return {
        "best_valid_rmsle": (
            None
            if math.isinf(best_score)
            else best_score
        ),
        "best_epoch": best_epoch,
        "last_epoch": max_epochs - 1,
        "global_step": global_step,
        "history": history
    }