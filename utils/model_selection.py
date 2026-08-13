from omegaconf import OmegaConf
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    KFold
)

def build_model_search(
    estimator,
    cfg_tuning,
    cfg_metric
):
    param_space = OmegaConf.to_container(
        cfg_tuning.param_space,
        resolve=True,
        throw_on_missing=True
    )

    # внутренний CV отвечает только за выбор гиперпараметров.
    inner_cv = KFold(
        n_splits=int(cfg_tuning.inner_cv.n_splits),
        shuffle=bool(cfg_tuning.inner_cv.shuffle),
        random_state=int(cfg_tuning.inner_cv.random_state)
    )

    common_params = {
        "estimator": estimator,
        "scoring": str(cfg_metric.sklearn_scoring),
        "cv": inner_cv,
        "n_jobs": int(cfg_tuning.n_jobs),
        "refit": True,
        "return_train_score": True,
        "error_score": "raise",
        "verbose": int(cfg_tuning.verbose),
    }

    search_type = str(cfg_tuning.search_type)

    if search_type == "grid":
        return GridSearchCV(
            param_grid=param_space,
            **common_params
        )
    elif search_type == "random":
        return RandomizedSearchCV(
            param_distributions=param_space,
            n_iter=int(cfg_tuning.n_iter),
            random_state=int(cfg_tuning.random_state),
            **common_params
        )

    raise ValueError(f"Unknown search type: {search_type}")