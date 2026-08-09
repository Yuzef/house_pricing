import numpy as np

from sklearn.ensemble import (
    RandomForestRegressor,
)
from sklearn.pipeline import Pipeline

from utils.load_data import load_data_func
from config import config

from utils.modeling import get_model_from_cfg
from utils.preprocessing import build_preprocessor
from utils.feature_engineering import build_feature_engineer

from utils.validation import run_cross_validation
from utils.modeling import get_model_from_cfg

from utils.experiment_artifacts import (
    prepare_experiment_dir,
    save_model,
)

from utils.experiment_logging import (
    save_cv_results,
    save_metrics,
    setup_experiment_logger,
)

def main() -> None:
    experiment_dir = prepare_experiment_dir(config)

    logger = setup_experiment_logger(experiment_dir)
    logger.info("Experiment started: %s", config.model.name)

    target_column = config.target.name
    id_column = config.id_column

    train_df, test_df = load_data_func(
        config.paths.train_csv,
        config.paths.test_csv,
    )

    logger.info(
        "Data loaded: train=%s, test=%s",
        train_df.shape,
        test_df.shape
    )

    X_train = train_df.drop(columns=[id_column, target_column])
    y_train = train_df[target_column]

    X_test = test_df.drop(columns=id_column)
    test_ids = test_df[id_column].copy()

    feature_engineer = build_feature_engineer(config.feature_engineering)

    preprocessor = build_preprocessor(config.preprocessing)
    
    model = get_model_from_cfg(config.model)

    pipeline = Pipeline(
        steps=[
            ("feature_engineering",  feature_engineer),
            ("preprocessing", preprocessor),
            ("model", model)
        ]
    )

    fold_results = run_cross_validation(
        estimator=pipeline,
        X=X_train,
        y=y_train,
        cfg_validation=config.validation,
        cfg_metric=config.metric,
    )

    print(fold_results)

    mean_val_score = fold_results["validation_score"].mean()
    std_val_score = fold_results["validation_score"].std()

    cv_results_path = save_cv_results(
        fold_results=fold_results,
        experiment_dir=experiment_dir
    )

    metrics_path = save_metrics(
        metric_name=config.metric.name,
        mean_score=mean_val_score,
        std_score=std_val_score,
        experiment_dir=experiment_dir
    )

    logger.info(
        "CV %s: %.5f +/- %.5f",
        config.metric.name,
        mean_val_score,
        std_val_score
    )

    logger.info(
        "CV results saved to: %s",
        cv_results_path
    )

    # cross_validate() обучает отдельные копии pipeline для folds.
    # Исходный объект pipeline после CV
    # не становится финальной обученной моделью.
    pipeline.fit(X_train,y_train)

    model_path = save_model(
        model=pipeline,
        experiment_dir=experiment_dir
    )

    print(f"Model saved to: {model_path}")

    test_predictions = pipeline.predict(X_test)





if __name__ == "__main__":
    main()
