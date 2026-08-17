from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from DL.checkpoints import read_checkpoint
from DL.data import to_float32_array
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
    X_test_transformed = to_float32_array(X_test_transformed)

    # Загрузка checkpoint и модели.
    checkpoint = read_checkpoint(
        path=checkpoint_path,
        device=device
    )
    
    model_params = dict(checkpoint["model_params"])

    if (
        X_test_transformed.shape[1] != model_params["input_dim"]
    ):
        raise ValueError(
            "Preprocessor output dimension does not "
            "match checkpoint input_dim."
        )
    
    model = HousePriceMLP(**model_params).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    # Inference.
    dataset = TensorDataset(torch.from_numpy(X_test_transformed))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    log_predictions = []

    with torch.inference_mode():
        for (features,) in loader:
            features = features.to(device)

            batch_predictions = model(features)

            log_predictions.append(batch_predictions.cpu().numpy())
        
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