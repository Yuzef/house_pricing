import numpy as np
import pandas as pd
import torch

from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader, TensorDataset

from utils.feature_engineering import build_feature_engineer
from utils.preprocessing import build_preprocessor
from utils.validation import build_cv_splits

def build_dl_feature_pipeline(config) -> Pipeline:
    """
    Отвечает только за преобразование raw DataFrame.
    """
    return Pipeline(
        steps = [
            (
                "feature_engineering",
                build_feature_engineer(
                    config.feature_engineering
                )
            ),
            (
                "preprocessing",
                build_preprocessor(
                    config.preprocessing
                )
            )
        ]
    )

def to_float32_array(features) -> np.ndarray:
    if hasattr(features, "toarray"):
        raise ValueError(
            "DL preprocessing returned a sparse matrix. "
            "Set nominal_encoding.sparse_output=False."
        )
    
    result = np.asarray(
        features,
        dtype=np.float32
    )

    if result.ndim !=2:
        raise ValueError(
            f"Expected 2D features, got shape {result.shape}."
        )
    
    return result

def transform_target(target: pd.Series) -> np.ndarray:
    """
    TransformedTargetRegressor в DL-ветке больше не используется.
    """
    values = target.to_numpy(dtype=np.float32)

    if np.any(values < 0):
        raise ValueError(
            "SalePrice contains negative values."
        )

    return np.log1p(values).astype(
        np.float32,
        copy=False
    )

def prepare_tuning_split(
    X: pd.DataFrame,
    y: pd.Series,
    config
):
    """
    Для tuning используется первый из 5 folds.
    """
    cv_splits = build_cv_splits(
        X=X,
        y=y,
        cfg_validation=config.validation
    )

    train_indices, valid_indices = cv_splits[0]

    feature_pipeline = build_dl_feature_pipeline(config)
