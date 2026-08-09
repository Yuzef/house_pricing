from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator

def create_submission(
    model,
    X_test,
    test_ids,
    id_column,
    prediction_column,
    experiment_dir,
    filename="submission.csv"
) -> Path:
    if len(X_test) != len(test_ids):
        raise ValueError(
            "X_test and test_ids must have the same number of rows."
        )

    # model - фактически полный обученный pipeline.
    predictions = model.predict(X_test)

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