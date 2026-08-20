from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

def average_predictions(predictions, weights=None):
    prediction_matrix = np.column_stack(predictions)

    if np.any(prediction_matrix < 0):
        raise ValueError(
            "SalePrice predictions must be non-negative."
        )

    averaged_prediction = np.average(
        prediction_matrix,
        axis=1,
        weights=weights,
    )

    return averaged_prediction

def create_submission(
    model,
    X_test,
    test_ids,
    id_column,
    prediction_column,
    experiment_dir,
    filename="submission.csv",
    predictions=None
) -> Path:
    if predictions is None:
        if model is None:
            raise ValueError(
                "Model must be provided when predictions are None."
            )
        
        # model - фактически полный обученный pipeline.
        predictions = model.predict(X_test)

    if len(predictions) != len(test_ids):
        raise ValueError(
            "Predictions and test_ids must have the same length."
        )


    submission = pd.DataFrame(
        {
            id_column: test_ids.to_numpy(),
            prediction_column: predictions
        }
    )

    if submission[prediction_column].isna().any():
        raise ValueError(
            "Submission contains missing predictions."
        )
    
    submission_path = experiment_dir / filename

    submission.to_csv(submission_path, index=False)

    return submission_path