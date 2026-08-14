import numpy as np
import pandas as pd
from sympy.logic.inference import valid
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

    # Разделение исходного X согласно индексам, которые дал CV.
    X_train_raw = X.iloc[train_indices]
    X_valid_raw = X.iloc[valid_indices]
    
    X_train = feature_pipeline.fit_transform(X_train_raw)
    X_valid = feature_pipeline.transform(X_valid_raw)

    X_train = to_float32_array(X_train)
    X_valid = to_float32_array(X_valid)

    y_log = transform_target(y)

    y_train = y_log[train_indices]
    y_valid = y_log[valid_indices]

    return {
        "X_train": X_train,
        "X_valid": X_valid,
        "y_train": y_train,
        "y_valid": y_valid,
        "train_indices": train_indices,
        "valid_indices": valid_indices,
        "feature_pipeline": feature_pipeline,
    }

def build_train_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool
):
    feature_tensor = torch.from_numpy(X)
    target_tensor = torch.from_numpy(y)

    dataset = TensorDataset(feature_tensor, target_tensor)

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        generator=generator if shuffle else None,
        persistent_workers=(num_workers>0)
    )

    return loader, generator