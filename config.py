from omegaconf import OmegaConf

config_dict = {
    'general': {
        "experiment_name": "25_pytorch_mlp_cosine_scheduler",
        "seed": 0xFACED,
        "task_type": "regression" 
    },
    "paths": {
        "train_csv": "data_raw/train.csv",
        "test_csv": "data_raw/test.csv",
        "trained_models": "trained_models"
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
        "n_jobs": 1, # 5 будет использовано при GridSearchCV.
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
            "type": "one_hot", # catboost_native or one_hot.
            "handle_unknown": "ignore",
            "sparse_output": False # False for DL pipeline.
        },

        "ordinal_encoding": {
            "enabled": False, # False for native catboost.
        },

        "scaling": {
            "enabled": True,
        },
    },
    
    "feature_engineering": {

        "age_features": {
            "enabled": False,
        },

        "area_features": {
            "enabled": False,
        },

        "bathroom_features": {
            "enabled": False,
        },

        "quality_area_interaction": {
            "enabled": True,
        },
    },

    "model": {
        "name": "25_pytorch_mlp_cosine_scheduler",
        "type": "DL",
        "params": {
            "batch_norm": {
                "enabled": False,
            }
        }
    },

    # sklearn tuning
    "tuning": {
        "enabled": False,
        "search_type": "random",
        "n_iter": 30,
        "random_state": "${general.seed}",

        "inner_cv": {
            "n_splits": 3,
            "shuffle": True,
            "random_state": "${general.seed}",
        },

        "n_jobs": 5,
        "verbose": 1,

        "param_space": {
            "model__regressor__n_estimators": [
                300,
                600,
                1000,
                1500,
            ],

            "model__regressor__learning_rate": [
                0.01,
                0.03,
                0.05,
                0.1,
            ],

            "model__regressor__max_depth": [
                2,
                3,
                4,
                6,
                8,
            ],

            "model__regressor__min_child_weight": [
                1.0,
                3.0,
                5.0,
                10.0,
            ],

            "model__regressor__gamma": [
                0.0,
                0.01,
                0.05,
                0.1,
            ],

            "model__regressor__subsample": [
                0.6,
                0.8,
                1.0,
            ],

            "model__regressor__colsample_bytree": [
                0.5,
                0.7,
                0.9,
                1.0,
            ],

            "model__regressor__reg_alpha": [
                0.0,
                0.001,
                0.01,
                0.1,
                1.0,
            ],

            "model__regressor__reg_lambda": [
                0.1,
                1.0,
                5.0,
                10.0,
                20.0,
            ],
        },
    },

    "dl": {
        "training": {
            "max_epochs": 100,
            "device": "cpu", # auto, mps not stable for AdamW
            "num_workers": 0,
            "pin_memory": False,
            "drop_last": False,
            "gradient_clip_norm": 1.0,
            "mixed_precision": False
        },

        "scheduler": {
            "enabled": True,
            "name": "cosine_annealing",
            "params": {
                # Сколько раз вызывать scheduler.
                # Здесь по 1 разу после каждой эпохи. 
                "T_max": "${dl.training.max_epochs}",
                "eta_min": 1e-6,
            }
        },
    },

    "optuna": {
        "study_name": "${model.name}",
        "target_n_trials": 15,
        "timeout_seconds": None,
        "n_jobs": 1,

        "inner_validation": {
            "strategy": "stratified_kfold",
            "n_splits": 3,
            "n_bins": 10,
            "shuffle": True,
            "random_state": "${general.seed}"
        },

        "sampler": {
            "name": "tpe",
            "seed": "${general.seed}",
            "n_startup_trials": 5
        },

        "pruner": {
            "name": "median",
            "n_startup_trials": 5,
            "n_warmup_steps": 1
        },

        "search_space": {
            "hidden_dim": [
                16,
                32,
                64
            ],

            "hidden_dim_2": [
                16,
                32,
            ],
            
            "activation": [
                "relu",
                "gelu",
                "silu"
            ],

            "batch_size": [
                16,
                32
            ],

            "learning_rate": {
                "adamw": {
                    "low": 0.0002,
                    "high": 0.001,
                    "log": True,
                },

                "adam": {
                    "low": 0.0002,
                    "high": 0.001,
                    "log": True,
                },

                "rmsprop": {
                    "low": 0.0001,
                    "high": 0.01,
                    "log": True,
                },
            },

            "weight_decay": {
                "low": 0.0000001,
                "high": 0.003,
                "log": True,
            },

            "dropout": [
                0.0
            ],

            "optimizer": [
                "adamw",
                "adam",
                "rmsprop",
            ]
        }
    },

    "inference": {
        "enabled": True,
        "prediction_column": "${target.name}",
        "submission_filename": "submission.csv",
    },

    "visualization": {
        "enabled": True,

        "save_fold_scores": True,
        "save_optuna_history": True,

        "save_dropout_effect": False,
        "save_optimizer_effect": True,

        "save_learning_rate_curve": True,

        "save_final_loss_curve": True,

        "show_plots": False,
        "figure_dpi": 150,
        "style": "seaborn-v0_8-whitegrid",
        "format": "png"
    }
  
}

config = OmegaConf.create(config_dict)