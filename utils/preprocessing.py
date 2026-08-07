import numpy as np

from sklearn.compose import (
    ColumnTransformer,
    make_column_selector,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

def build_preprocessor(cfg_preprocessing) -> ColumnTransformer:
#             исходный X
#                │
#        ┌───────┴────────┐
#        ↓                ↓
#    numerical        categorical
#        │                │
#     imputer          imputer
#        │                │
#     scaler           one-hot
#        │                │
#        └───────┬────────┘
#                ↓
#          готовый X
#                ↓
#              model

    # pipeline для числовых признаков.
    numerical_steps = [
        (
            "imputer", # имя шага. Filling missing value.
            SimpleImputer(
                strategy=(cfg_preprocessing.numerical_imputation.strategy),
            ),
        ),
    ]

    if cfg_preprocessing.scaling.enabled:
        numerical_steps.append("scaler", StandardScaler())

    numerical_pipeline = Pipeline(
        steps=numerical_steps
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy=(
                        cfg_preprocessing
                        .categorical_imputation
                        .strategy
                    ),
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown=(
                        cfg_preprocessing.nominal_encoding.handle_unknown
                    ),
                    sparse_output=True,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                make_column_selector(dtype_include=np.number)
            ),
            (
                "categorical",
                categorical_pipeline,
                make_column_selector(dtype_include=["object","category"])
            ),
        ],
        remainder="drop", # все колонки, которые не попали ни в один selector, удалить.
        verbose_feature_names_out=False, # не добавлять numerical__ categorical__
                                         # к названиями колонок.
    )
