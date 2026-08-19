from torch.optim.lr_scheduler import CosineAnnealingLR


def build_scheduler(
    *,
    optimizer,
    cfg_scheduler,
):
    if not bool(cfg_scheduler.enabled):
        return None

    scheduler_name = str(cfg_scheduler.name).lower()

    if scheduler_name != "cosine_annealing":
        raise ValueError(
            f"Unknown scheduler: {scheduler_name}."
        )

    t_max = int(cfg_scheduler.params.T_max)
    eta_min = float(cfg_scheduler.params.eta_min)

    if t_max < 1:
        raise ValueError("Scheduler T_max must be positive.")

    if eta_min < 0.0:
        raise ValueError("Scheduler eta_min cannot be negative.")

    return CosineAnnealingLR(
        optimizer=optimizer,
        T_max=t_max,
        eta_min=eta_min,
    )