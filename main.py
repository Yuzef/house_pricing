from sklearn.pipeline import Pipeline

from utils.load_data import load_data_func
from config import config

from utils.modeling import get_model_from_cfg
from utils.preprocessing import build_preprocessor
from utils.feature_engineering import build_feature_engineer

from utils.validation import run_cross_validation
from utils.model_selection import build_model_search

from utils.inference import create_submission
from DL.workflow import run_dl_experiment

from utils.experiment_artifacts import (
    prepare_experiment_dir,
    save_model,
)

from utils.experiment_logging import (
    save_best_params,
    save_cv_results,
    save_search_results,
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

    # Развилка.

    # DL ветка.
    if str(config.model.type) == "DL":
        run_dl_experiment(
            config=config,
            experiment_dir=experiment_dir,
            logger=logger,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            test_ids=test_ids
        )

        logger.info("DL experiment comleted successfully")

        return
    
    # Классическая sklearn ветка.
    feature_engineer = build_feature_engineer(config.feature_engineering)

    encoding_type = str(config.preprocessing.nominal_encoding.type)

    if encoding_type == "catboost_native":
        if str(config.model.type) != "catboost":
            raise ValueError(
                "catboost_native encoding can only "
                "be used with the CatBoost model."
            )
        
        cat_features = tuple(
            X_train.select_dtypes(
                include=["object", "category"]
            ).columns
        )
    
    else:
        cat_features = None

    preprocessor = build_preprocessor(config.preprocessing)
    
    model = get_model_from_cfg(config.model, cat_features=cat_features)

    pipeline = Pipeline(
        steps=[
            ("feature_engineering",  feature_engineer),
            ("preprocessing", preprocessor),
            ("model", model)
        ]
    )

    if config.tuning.enabled:
        estimator_for_cv = build_model_search(
            estimator=pipeline,
            cfg_tuning=config.tuning,
            cfg_metric=config.metric
        )
    else:
        estimator_for_cv = pipeline

    fold_results = run_cross_validation(
        estimator=estimator_for_cv,
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
        "Metrics saved to: %s",
        metrics_path,
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
    if config.tuning.enabled:
        model_search = build_model_search(
            estimator=pipeline,
            cfg_tuning=config.tuning,
            cfg_metric=config.metric
        )

        model_search.fit(X_train, y_train)

        search_results_path = save_search_results(
            search=model_search,
            experiment_dir=experiment_dir
        )

        best_params_path = save_best_params(
            best_params=model_search.best_params_,
            experiment_dir=experiment_dir
        )

        final_pipeline = model_search.best_estimator_

        logger.info(
            "Best parameters: %s",
            model_search.best_params_
            )

        logger.info(
            "Best inner CV RMSLE: %.5f",
            -model_search.best_score_,
        )
    
    else:
        pipeline.fit(X_train, y_train)
        final_pipeline = pipeline

    model_path = save_model(
        model=final_pipeline,
        experiment_dir=experiment_dir
    )

    logger.info(
        "Model saved to: %s",
        model_path,
    )

    if config.inference.enabled:
        submission_path = create_submission(
            model=final_pipeline,
            X_test=X_test,
            test_ids=test_ids,
            id_column=config.id_column,
            prediction_column=config.inference.prediction_column,
            experiment_dir=experiment_dir,
            filename=config.inference.submission_filename
        )
    
        logger.info("Submission saved to: %s", submission_path)

    logger.info("Experiment completed successfully")


if __name__ == "__main__":
    main()
