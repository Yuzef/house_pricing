from omegaconf import OmegaConf

config_dict = {
    'general': {
        "experiment_name": "Boosting",
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
        "name": "19_pytorch_mlp_optuna",
        "type": "DL",

        "params": {
            "hidden_dim": 128,
            "activation": "relu",
        },
    },

    "dl": {
        "training": {
            "max_epochs": 300,
            "device": "auto",
            "num_workers": 0,
            "pin_memore": False,
            "gradient_clip_norm": 1.0,
            "mixed_precision": False,
        }
    },

    "optuna": {
        "study_name": "${model.name}",
        "target_n_trials": 30,
        "timeout_seconds": None,
        "n_jobs": 1,

        "sampler": {
            "name": "tpe",
            "seed": "${general.seed}"
        },

        "pruner": {
            "name": "median",
            "n_startup_trials": 5,
            "n_warmup_steps": 20
        },

        "search_space": {
            "hidden_dim": [
                32,
                64,
                128,
                256
            ],
            
            "activation": [
                "relu",
                "gelu",
                "silu"
            ],

            "batch_size": [
                32,
                64,
                128
            ],

            "learning_rate": {
                "low": 0.0001,
                "high": 0.003,
                "log": True,
            },

            "weight_decay": {
                "low": 0.0000001,
                "high": 0.01,
                "log": True,
            },
        }
    }
   
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