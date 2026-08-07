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

    "validation": {
        "strategy": "stratified_kfold",
        "n_splits": 5,
        "n_bins": 10,
        "shuffle": True,
        "target_column": "SalePrice",
    },
    "preprocessing": {
        # Заполнение пропусков.
        "numerical_imputation": {
                "strategy": "median",
        },
        "categorical_imputation": {
                "strategy": "most_frequent",
        },











        "embarked": {
            "enabled": True,
            "strategy": "most_frequent",
        },
        "age": {
            "enabled": True,
            "strategy": "mean_by_title"
        },
        "initial": {
            "enabled": True,
            "output_column": "Initial",
        },
        "age_binning": {
            "enabled": True,
            "strategy": "equal_width",   # "quantile"
            "output_column": "Age_band",
            "num_bins": 5,
            "drop_original": True,
        },
        "categorical_encoding": {
            "enabled": True,

            "use_embedding": False,
            "use_one_hot_encoding": True,

            # Одни и те же категориальные признаки
            # либо кодируем one-hot, либо передаём в Embedding.
            "columns": [
                "Embarked",
                "Initial",
            ],

            
            "mapping": {
                "enabled": True, # False for pure catboost native categorical.
                "columns": {
                    "Sex": {
                        "male": 0,
                        "female": 1,
                    },
                },
            },
            # Используется только при one_hot_encoding.
            "one_hot_params": {
                "drop_first": True,   # для LogisticRegression можно попробовать
                                      # поставить True
            },
        },
        "family_features": {
            "enabled": True,
            "family_size_column": "Family_Size",
            "alone_column": "Alone",
            "drop_original": True,      # удалять ли потом SibSp и Parch
                                        # (KNN, LogRef - True, деревья - False.)
        },
        "fare": { # если в тесте будет пропуск, то заменяем его значением median из train.
            "enabled": True,
            "strategy": "median",
        },
        "fare_binning": {
            "enabled": True,
            "strategy": "quantile",
            "output_column": "Fare_Range",
            "num_bins": 4,
            "drop_original": True,
        },
        "features": {
            "given_columns": [ # что изначально дали?
                "PassengerId",
                "Survived",
                "Pclass",
                "Name",
                "Sex",
                "Age",     # Исходная колонка до FE
                "SibSp",   # Исходная колонка
                "Parch",   # Исходная колонка
                "Ticket",  # Удаляем
                "Fare",    # Исходная колонка до FE
                "Cabin",   # Удаляем
                "Embarked",
            ],
            "use_columns": [    # что используем?
                "Pclass",
                "Sex",
                # "Age",        # drop original ?
                "Age_band",     # feature engineering
                # "SibSp",      # drop original ?
                # "Parch",      # drop original ?
                # "Fare",       # drop original ?
                "Fare_Range",   # feature engineering
                "Family_Size",  # feature engineering
                "Alone",        # feature engineering
                # "Embarked",   # categorical_encoding
                # "Initial",    # categorical_encoding
            ],
          
            # "cat_features": [
            #     "Sex", "Embarked", "Initial",
            #     ]

        },
    },
    "modeling": {
        # Приведение всех числовых признаков к одному масштабу.
        # StandardScaler(): x_scaled = (x - mean) / std
        # LogReg - True, KNN - True, RandomForest - False, Boosting - False
        "scale_features": False, # True для DL, False для бустингов.
        # Использовать все доступные ядра процессора "-1".
        #n_jobs=6 — использовать ровно 6 ядер.
        "n_jobs": 6,

        "models": [
            {
                "name": f"rf_100_depth_{depth}_leaf_{leaf}",
                "enabled": True,
                "type": "random_forest",
                "params": {
                    "n_estimators": 100,
                    "max_depth": depth,
                    "min_samples_split": 2,
                    "min_samples_leaf": leaf,
                    "max_features": "sqrt",
                    "bootstrap": True,
                },
            }
            for depth in range(4, 8)
            for leaf in range(1, 5)
        ]
    },
    "dl": {
        "training": {
            "num_epochs": 150,
            "full_train_epochs": "auto", # Берётся лучшее количество epochs,
                                         # определённое по CV.
            "epoch_selection_metric": "valid_accuracy",
            "epoch_selection_mode": "max",
            # Рассчет метрик на валидации раз в n epochs.
            "eval_every_n_epochs": 1,
            "device": "auto", # код сам выберет cuda / mps / cpu
            "mixed_precision": False, # подключим после BL запуска.
            "verbose": False,
            # "early_stopping_epochs": 5,
            "lr": 8e-4,
            
        },
        "checkpoint":{
            "enabled": True,
            "monitor": "valid_accuracy", #по какой метрике выбирать best checkpoint
            "mode": "max", # max для accuracy, min для loss
            "save_optimizer_state": True, # нужно ли уметь продолжить обучение
            "save_best": True,
            "save_last": False,
            "extension": "pt",
        },
        "dataloader_params": {
            "batch_size": 32,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "shuffle": True,
            # Если последний batch получился неполным, он будет отброшен.
            "drop_last": False, # Ставлю False, т.к. dataset маленький.
        },
        "optimizer": {
            "name": "adam",
            "params": {
                "lr": "${dl.training.lr}",
                # инерция оптимизатора
                # "momentum": 0.9,
                # Регуляризация.
                "weight_decay": 0.0001,
            },
        },
        "scheduler": {
            "enabled": True,
            "name": "cosine_annealing",
            "params": {
                # Сколько раз вызывать scheduler.
                # Здесь по 1 разу после каждой эпохи. 
                "T_max": "${dl.training.num_epochs}",
                "eta_min": 1e-6,
            }
        },
        "loss": {
            # CrossEntropyLoss потому что он подойдёт потом под расширение
            # на многоклассовую классификацию - более универсальный pipeline,
            # но надо помнить, что вычислительно CrossEntropyLoss сложнее для
            # бинарной классификации, так как он считает 2 logits, а не один. 
            "name": "cross_entropy",
            "params": {
                "label_smoothing": 0.05,
            }
        },
    },

    "metric": {
        "name": "RMSE"
    },
    "inference": {
        "enabled": True,
        "model_name": "rf_100_depth_5_leaf_3",  # если выбрать inference вручную
                                                # по названию .joblib файла.
        "use_best_model": True, # выберет _BEST .joblib 
        "id_column": "PassengerId",
        "prediction_column": "Survived",
        "submission_dir": "submissions",
    },
    "logging": {
        "enabled": True,
        "save_config": True,
        "save_fold_results": True,
        "save_summary": True,
        "save_artifact_paths": True, 
        "save_best_model": True,
        "save_readable_report": True,
    },
    "visualization": {
        "enabled": True,

        "save_training_curves": True,
        "save_fold_scores": True,
        "save_summary_bar": True,

        "show_plots": False,
        "figure_dpi": 150,
        "style": "whitegrid",

        "training_curves": {
            "metrics": ["loss", "accuracy"],
            "plot_train": True,
            "plot_valid": True,
        },


    }
}

config = OmegaConf.create(config_dict)