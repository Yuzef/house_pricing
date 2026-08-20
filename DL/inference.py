from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from DL.checkpoints import read_checkpoint
from DL.data import (
    split_embedding_features,
    to_float32_array,
)
from DL.dl_models.mlp import HousePriceMLP

def create_dl_submission(
    *,
    checkpoint_path: Path,
    feature_pipeline_path: Path,
    X_test,
    test_ids,
    id_column: str,
    prediction_column: str,
    output_path: Path,
    device,
    batch_size: int
) -> Path:

    # Загрузка preprocessing.
    feature_pipeline = joblib.load(feature_pipeline_path)
    X_test_transformed = feature_pipeline.transform(X_test)

    checkpoint = read_checkpoint(path=checkpoint_path, device=device)

    model_params = dict(checkpoint["model_params"])

    use_embeddings = model_params.get("categorical_cardinalities")is not None
    
    if not use_embeddings:
        X_test_transformed = to_float32_array(X_test_transformed)

        if (
            X_test_transformed.shape[1]
            != model_params["input_dim"]
        ):
            raise ValueError(
                "Preprocessor output dimension does "
                "not match checkpoint input_dim."
            )

        dataset = TensorDataset(torch.from_numpy(X_test_transformed))

    else:
        (
            X_test_numerical,
            X_test_categorical,
            categorical_cardinalities,
        ) = split_embedding_features(
            X_test_transformed,
            feature_pipeline,
        )

        if (
            list(categorical_cardinalities)
            != model_params["categorical_cardinalities"]
        ):
            raise ValueError(
                "Categorical cardinalities do not "
                "match checkpoint."
            )

        if (
            X_test_numerical.shape[1]
            != model_params["numerical_dim"]
        ):
            raise ValueError(
                "Numerical dimension does not "
                "match checkpoint."
            )

        dataset = TensorDataset(
            torch.from_numpy(X_test_numerical),
            torch.from_numpy(X_test_categorical)
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    log_predictions = []

    with torch.inference_mode():
        for batch in loader:
            if len(batch) == 1:
                features = batch[0].to(device)

                batch_predictions = model(
                    features
                )

            elif len(batch) == 2:
                numerical_features = batch[0].to(
                    device
                )

                categorical_features = batch[1].to(
                    device
                )

                batch_predictions = model(
                    numerical_features,
                    categorical_features,
                )

            else:
                raise ValueError(
                    f"Unexpected inference batch: "
                    f"{len(batch)}."
                )

            log_predictions.append(
                batch_predictions.cpu().numpy()
            )
        
    # Обратное преобразование.
    log_predictions = np.concatenate(log_predictions)

    predictions = np.expm1(log_predictions)
    predictions = np.clip(
        predictions,
        a_min=0.0,
        a_max=None
    )

    # Submission.
    submission = pd.DataFrame(
        {
            id_column: test_ids.to_numpy(),
            prediction_column: predictions
        }
    )

    if submission[prediction_column].isna().any():
        raise ValueError("Submission contanins NaN predictions.")
    
    submission.to_csv(output_path, index=False)

    return output_path