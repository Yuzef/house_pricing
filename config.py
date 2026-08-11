from omegaconf import OmegaConf

config_dict = {
    'general': {
        "experiment_name": "Baseline",
        "seed": 0xFACED,
        "task_type": "regression" 
    },
    "paths": {
        "train_csv": "data_raw/train.csv",
        "test_csv": "data_raw/test.csv",
        "trained_models": "trained_models",
        "plots_dir": "plots",
    },
    "target": {
        "name": "SalePrice",
        "transform": "log1p"
    },
    "metric": {
        "name": "RMSLE", # TransformedTargetRegressor возвращает предсказания
                        # обратно в долларах.
        "sklearn_scoring": "neg_root_mean_squared_log_error"
    },

    "id_column": "Id",

    "validation": {
        "strategy": "stratified_kfold",
        "n_splits": 5,
        "n_bins": 10,
        "shuffle": True,
        "random_state": "${general.seed}",
        "n_jobs": 6,
        "return_train_score": True
    },

    "preprocessing": {
        # Заполнение пропусков.
        "numerical_imputation": {
                "strategy": "median",
        },

        "categorical_imputation": {
                "strategy": "most_frequent",
        },

        "nominal_encoding": {
            "type": "one_hot",
            "handle_unknown": "ignore",
        },

        "ordinal_encoding": {
            "enabled": False,
        },

        "scaling": {
            "enabled": True,
        },
    },
    
    "feature_engineering": {

        "age_features": {
            "enabled": True,
        },

        "area_features": {
            "enabled": False,
        },

        "bathroom_features": {
            "enabled": False,
        },

        "quality_area_interaction": {
            "enabled": False,
        },
    },

    "model": {
        "name": "2_lin_reg_BL_age_features",
        "type": "linear_regression",
        "params": {
            "fit_intercept": True, # модель сама вычисляет b0
        },

            # random forest

            # "name": f"rf_100_depth_6_leaf_2",
            # "type": "random_forest",
            # "params": {
            #     "n_estimators": 100,
            #     "max_depth": 6,
            #     "min_samples_split": 2,
            #     "min_samples_leaf": 2,
            #     "max_features": "sqrt",
            #     "bootstrap": True,
            #     "random_state": "${general.seed}",
            #     "n_jobs": 1,
            # },
            # "ridge":{
            #     "params":{
            #         "alpha": 10.0
            #     }
            # }
        },

    # "dl": {
    #     "training": {
    #         "num_epochs": 150,
    #         "full_train_epochs": "auto", # Берётся лучшее количество epochs,
    #                                      # определённое по CV.
    #         "epoch_selection_metric": "valid_accuracy",
    #         "epoch_selection_mode": "max",
    #         # Рассчет метрик на валидации раз в n epochs.
    #         "eval_every_n_epochs": 1,
    #         "device": "auto", # код сам выберет cuda / mps / cpu
    #         "mixed_precision": False, # подключим после BL запуска.
    #         "verbose": False,
    #         # "early_stopping_epochs": 5,
    #         "lr": 8e-4,
            
    #     },
    #     "checkpoint":{
    #         "enabled": True,
    #         "monitor": "valid_accuracy", #по какой метрике выбирать best checkpoint
    #         "mode": "max", # max для accuracy, min для loss
    #         "save_optimizer_state": True, # нужно ли уметь продолжить обучение
    #         "save_best": True,
    #         "save_last": False,
    #         "extension": "pt",
    #     },
    #     "dataloader_params": {
    #         "batch_size": 32,
    #         "num_workers": 0,
    #         "pin_memory": False,
    #         "persistent_workers": False,
    #         "shuffle": True,
    #         # Если последний batch получился неполным, он будет отброшен.
    #         "drop_last": False, # Ставлю False, т.к. dataset маленький.
    #     },
    #     "optimizer": {
    #         "name": "adam",
    #         "params": {
    #             "lr": "${dl.training.lr}",
    #             # инерция оптимизатора
    #             # "momentum": 0.9,
    #             # Регуляризация.
    #             "weight_decay": 0.0001,
    #         },
    #     },
    #     "scheduler": {
    #         "enabled": True,
    #         "name": "cosine_annealing",
    #         "params": {
    #             # Сколько раз вызывать scheduler.
    #             # Здесь по 1 разу после каждой эпохи. 
    #             "T_max": "${dl.training.num_epochs}",
    #             "eta_min": 1e-6,
    #         }
    #     },
    #     "loss": {
    #         # CrossEntropyLoss потому что он подойдёт потом под расширение
    #         # на многоклассовую классификацию - более универсальный pipeline,
    #         # но надо помнить, что вычислительно CrossEntropyLoss сложнее для
    #         # бинарной классификации, так как он считает 2 logits, а не один. 
    #         "name": "cross_entropy",
    #         "params": {
    #             "label_smoothing": 0.05,
    #         }
    #     },
    # },

    "inference": {
        "enabled": True,
        "prediction_column": "${target.name}",
        "submission_filename": "submission.csv",
    },

    # "visualization": {
    #     "enabled": True,

    #     "save_training_curves": True,
    #     "save_fold_scores": True,
    #     "save_summary_bar": True,

    #     "show_plots": False,
    #     "figure_dpi": 150,
    #     "style": "whitegrid",

    #     "training_curves": {
    #         "metrics": ["loss", "accuracy"],
    #         "plot_train": True,
    #         "plot_valid": True,
    #     },
    # }
}

config = OmegaConf.create(config_dict)