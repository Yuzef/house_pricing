import matplotlib.pyplot as plt

from pathlib import Path

from optuna.visualization.matplotlib import (
    plot_optimization_history,
)

from optuna.visualization.matplotlib import plot_slice

def plot_outer_fold_scores(fold_results, output_path, cfg):
    scores = fold_results["validation_score"]
    folds = fold_results["fold"]

    mean_score = scores.mean()
    std_score = scores.std()

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.scatter(
        folds,
        scores,
        s=70,
        color="tab:blue",
        zorder=3,
        label="Outer fold RMSLE",
    )

    ax.axhline(
        mean_score,
        color="tab:red",
        linestyle="--",
        label=f"Mean: {mean_score:.4f}",
    )

    ax.fill_between(
        [folds.min() - 0.2, folds.max() + 0.2],
        mean_score - std_score,
        mean_score + std_score,
        color="tab:red",
        alpha=0.12,
        label=f"±1 std: {std_score:.4f}",
    )

    for fold, score in zip(folds, scores):
        ax.annotate(
            f"{score:.4f}",
            (fold, score),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
        )

    ax.set(
        title="Nested CV: outer-fold scores",
        xlabel="Outer fold",
        ylabel="RMSLE (lower is better)",
    )
    ax.set_xticks(folds)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=int(cfg.figure_dpi),
        bbox_inches="tight",
    )

    if cfg.show_plots:
        plt.show()

    plt.close(fig)

def plot_optuna_history(study, output_path, cfg):
    ax = plot_optimization_history(study)

    ax.set_title("Optuna optimization history")
    ax.set_ylabel("Inner CV RMSLE")

    fig = ax.figure
    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=int(cfg.figure_dpi),
        bbox_inches="tight",
    )

    if cfg.show_plots:
        plt.show()

    plt.close(fig)

def plot_dropout_effect(study, output_path, cfg):
    ax = plot_slice(
        study,
        params=["dropout"],
    )

    ax.set_title("Dropout vs inner CV RMSLE")
    ax.set_ylabel("Inner CV RMSLE (lower is better)")

    fig = ax.figure
    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=int(cfg.figure_dpi),
        bbox_inches="tight",
    )

    if cfg.show_plots:
        plt.show()

    plt.close(fig)

def plot_optimizer_effect(study, output_path, cfg):
    ax = plot_slice(
        study,
        params=["optimizer"],
    )

    ax.set_title("Optimizer vs inner CV RMSLE")
    ax.set_ylabel("Inner CV RMSLE (lower is better)")

    fig = ax.figure
    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=int(cfg.figure_dpi),
        bbox_inches="tight",
    )

    if cfg.show_plots:
        plt.show()

    plt.close(fig)

def plot_final_loss_curve(history, output_path, cfg):
    if not history:
        raise ValueError("Training history is empty.")

    epochs = [
        int(epoch_record["epoch"]) + 1
        for epoch_record in history
    ]

    train_losses = [
        float(epoch_record["train_loss"])
        for epoch_record in history
    ]

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(
        epochs,
        train_losses,
        color="tab:blue",
        linewidth=2,
        label="Train loss",
    )

    ax.set(
        title="Final model training loss",
        xlabel="Epoch",
        ylabel="MSE loss in log1p target space",
    )

    # В начале loss значительно больше, чем в конце.
    # Логарифмическая шкала позволит увидеть
    # весь процесс обучения.
    ax.set_yscale("log")

    ax.legend()
    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=int(cfg.figure_dpi),
        bbox_inches="tight",
    )

    if cfg.show_plots:
        plt.show()

    plt.close(fig)

def create_experiment_plots(
    *,
    fold_results,
    study,
    final_history,
    experiment_dir: Path,
    config,
) -> list[Path]:

    if not bool(config.visualization.enabled):
        return []

    plt.style.use(str(config.visualization.style))

    plots_dir = experiment_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    extension = str(config.visualization.format)
    created_paths = []

    if config.visualization.save_fold_scores:
        path = plots_dir / f"outer_fold_scores.{extension}"

        plot_outer_fold_scores(
            fold_results=fold_results,
            output_path=path,
            cfg=config.visualization,
        )
        
        created_paths.append(path)

    if config.visualization.save_final_loss_curve:
        path = plots_dir / f"final_training_loss.{extension}"

        plot_final_loss_curve(
            history=final_history,
            output_path=path,
            cfg=config.visualization,
        )

        created_paths.append(path)

    if config.visualization.save_optuna_history:
        path = plots_dir / f"optuna_history.{extension}"

        plot_optuna_history(
            study=study,
            output_path=path,
            cfg=config.visualization,
        )

        created_paths.append(path)

    if config.visualization.save_dropout_effect:
        path = plots_dir / f"dropout_effect.{extension}"

        plot_dropout_effect(
            study=study,
            output_path=path,
            cfg=config.visualization,
        )

        created_paths.append(path)
    
    if config.visualization.save_optimizer_effect:
        path = plots_dir / f"optimizer_effect.{extension}"

        plot_optimizer_effect(
            study=study,
            output_path=path,
            cfg=config.visualization,
        )

        created_paths.append(path)

    return created_paths
    