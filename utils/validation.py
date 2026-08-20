'''
Для проекта используется:

`StratifiedKFold`

- `n_splits = 5`;
- `shuffle = True`;
- фиксированным `random_state`;
- стратификацией по квантильным диапазонам `SalePrice`.

Поскольку задача является регрессией, непосредственно передать
непрерывный `SalePrice` в `StratifiedKFold` нельзя.

Поэтому таргет временно разбивается на ценовые диапазоны.

Эти диапазоны используются только для создания folds и не являются
новым target.
'''

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate as sklearn_cross_validate,
)

def build_cv_splits(
    X: pd.DataFrame,
    y: pd.Series,
    cfg_validation
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Построить стратифицированные фолды для задачи регрессии.

    Для использования StratifiedKFold непрерывный таргет временно
    разбивается на квантильные ценовые диапазоны. Полученные категории
    используются только для формирования фолдов.

    Returns:
        Список пар `(train_indices, validation_indices)`.
    """
    if cfg_validation.strategy != "stratified_kfold":
        raise ValueError(
            f"Unsupported validation strategy: "
            f"{cfg_validation.strategy}"
        )
    
    n_splits = int(cfg_validation.n_splits)
    n_bins = int(cfg_validation.n_bins)
    shuffle = bool(cfg_validation.shuffle)

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows")
    if y.isna().any():
        raise ValueError("Target contains missing values")
    
    stratification_labels = pd.qcut( # pd.qcut() старается сделать примерно
                                     # одинаковое кол-во объектов в каждой группе.
        y,
        q=n_bins,
        labels=False,
        duplicates="drop"
    )

    actual_n_bins = (
        stratification_labels.nunique()
    )

    if actual_n_bins != n_bins:
        raise ValueError(
            f"Requested {n_bins} target bins, "
            f"but only {actual_n_bins} could "
            "be created"
        )

    bin_counts = stratification_labels.value_counts()
    # Пр-ка возможно ли распределить класс так по бинам,
    # чтобы он присутствовал в каждом.
    # Для этого объектов в бине должно быть больше чем самих бинов. 
    if bin_counts.min() < n_splits:
        raise ValueError(
            "Every target bin must contain at least n_splits observations. "
            f"Smallest bin: {bin_counts.min()}, n_splits: {n_splits}."
        )
    
    random_state = (
        int(cfg_validation.random_state)
        if shuffle
        else None
    )

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state
    )
    # StratifiedKFold ожидает классы
    return list(splitter.split(X, stratification_labels))

def run_cross_validation(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    cfg_validation,
    cfg_metric
) -> pd.DataFrame:

    cv_splits = build_cv_splits(
        X=X,
        y=y,
        cfg_validation=cfg_validation
    )

    scoring = str(cfg_metric.sklearn_scoring)

    return_train_score = bool(cfg_validation.return_train_score)

    raw_results = sklearn_cross_validate(
        estimator=estimator,
        X=X,
        y=y,
        cv=cv_splits,
        scoring=scoring,
        n_jobs=int(cfg_validation.n_jobs),
        return_train_score=return_train_score,
        error_score="raise"
    )

    score_multiplier = (
        -1.0
        if scoring.startswith("neg_")
        else 1.0
    )

    fold_results = pd.DataFrame(
        {
            "fold": np.arange(
                1,
                len(raw_results["test_score"]) + 1,
            ),
            "validation_score": (
                score_multiplier * raw_results["test_score"]
            ),
            "fit_time_seconds": raw_results["fit_time"],
            "score_time_seconds": raw_results["score_time"]
        }
    )

    if return_train_score:
        fold_results["train_score"] = (
            score_multiplier * raw_results["train_score"]
        )
    
    return fold_results

