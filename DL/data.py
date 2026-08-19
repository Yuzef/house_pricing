import numpy as np
import pandas as pd
import torch

from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader, TensorDataset

from omegaconf import OmegaConf
from utils.feature_engineering import build_feature_engineer
from utils.preprocessing import build_preprocessor

def embeddings_enabled(config) -> bool:
    return (
        str(
            config
            .preprocessing
            .nominal_encoding
            .type
        )
        == "embedding"
    )

def split_embedding_features(
    features,
    feature_pipeline,
):
    if hasattr(features, "toarray"):
        raise ValueError(
            "Embedding preprocessing returned "
            "a sparse matrix."
        )

    features = np.asarray(features)

    preprocessor = (
        feature_pipeline
        .named_steps["preprocessing"]
    )

    output_indices = preprocessor.output_indices_

    numerical_parts = [
        np.asarray(
            features[:, output_indices["numerical"]],
            dtype=np.float32,
        )
    ]

    # Если ordinal quality когда-нибудь будет включён,
    # эти признаки остаются числовыми.
    if "ordinal_quality" in output_indices:
        ordinal_slice = output_indices[
            "ordinal_quality"
        ]

        if ordinal_slice.stop > ordinal_slice.start:
            numerical_parts.append(
                np.asarray(
                    features[:, ordinal_slice],
                    dtype=np.float32,
                )
            )

    numerical_features = np.concatenate(
        numerical_parts,
        axis=1,
    )

    categorical_features = np.asarray(
        features[:, output_indices["categorical"]],
        dtype=np.int64,
    )

    # unknown -1 становится индексом 0.
    categorical_features += 1

    encoder = (
        preprocessor
        .named_transformers_["categorical"]
        .named_steps["encoder"]
    )

    categorical_cardinalities = [
        len(categories) + 1
        for categories in encoder.categories_
    ]

    return (
        numerical_features,
        categorical_features,
        categorical_cardinalities,
    )

def split_dl_features(features, feature_pipeline):
    features = np.asarray(features)

    preprocessor = feature_pipeline.named_steps["preprocessing"]
    output_indices = preprocessor.output_indices_

    numerical = np.asarray(
        features[:, output_indices["numerical"]],
        dtype=np.float32,
    )

    categorical = np.asarray(
        features[:, output_indices["categorical"]],
        dtype=np.int64,
    )

    categorical += 1  # unknown -1 становится индексом 0

    encoder = (
        preprocessor
        .named_transformers_["categorical"]
        .named_steps["encoder"]
    )

    categorical_cardinalities = [
        len(categories) + 1  # +1 для unknown
        for categories in encoder.categories_
    ]

    return numerical, categorical, categorical_cardinalities

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

def prepare_fold(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    train_indices,
    valid_indices,
    config
):
    feature_pipeline = build_dl_feature_pipeline(config)

    # Разделение исходного X согласно индексам, которые дал CV.
    X_train_raw = X.iloc[train_indices]
    X_valid_raw = X.iloc[valid_indices]
    
    X_train = feature_pipeline.fit_transform(X_train_raw)
    X_valid = feature_pipeline.transform(X_valid_raw)

    if embeddings_enabled(config):
        (
            X_train,
            X_cat_train,
            categorical_cardinalities
        ) = split_embedding_features(
            X_train,
            feature_pipeline
        )

        (
            X_valid,
            X_cat_valid,
            valid_cardinalities
        ) = split_embedding_features(
            X_valid,
            feature_pipeline
        )

        if (
            categorical_cardinalities != valid_cardinalities
        ):
            raise ValueError(
            "Train and validation cardinalities "
            "do not match."
            )
    
    else:

        X_train = to_float32_array(X_train)
        X_valid = to_float32_array(X_valid)

        X_cat_train = None
        X_cat_valid = None
        categorical_cardinalities = None

    y_train = transform_target(y.iloc[train_indices])
    y_valid = transform_target(y.iloc[valid_indices])

    return {
        "X_train": X_train,
        "X_valid": X_valid,

        "X_cat_train": X_cat_train,
        "X_cat_valid": X_cat_valid,

        "categorical_cardinalities": (categorical_cardinalities),

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
    drop_last: bool,
    X_cat: np.ndarray | None = None
):
    feature_tensor = torch.from_numpy(X)
    target_tensor = torch.from_numpy(y)

    if X_cat is None:
        dataset = TensorDataset(
            feature_tensor,
            target_tensor
        )
    
    else:
        categorical_tensor = torch.from_numpy(
            X_cat
        )

        if categorical_tensor.dtype != torch.long:
            raise TypeError(
            "Categorical tensor must have "
            "torch.long dtype."
            )
        
        dataset = TensorDataset(
            feature_tensor,
            categorical_tensor,
            target_tensor
        )




    # Генератор случайных чисел,
    # фиксируем порядок перемешивания для воспроизводимости.
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