import numpy as np

from sklearn.compose import (
    ColumnTransformer,
    make_column_selector,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

ORDINAL_QUALITY_COLUMNS = (
    "ExterQual",
    "ExterCond",
    "BsmtQual",
    "BsmtCond",
    "HeatingQC",
    "KitchenQual",
    "FireplaceQu",
    "GarageQual",
    "GarageCond",
)

QUALITY_ORDER = (
    "NA",
    "Po",
    "Fa",
    "TA",
    "Gd",
    "Ex",
)

def select_nominal_columns(X):
    categorical_columns = (
        X.select_dtypes(include=[
            "object",
            "category"
        ]).columns
    )

    return [
        column for column in categorical_columns
        if column not in ORDINAL_QUALITY_COLUMNS
    ]

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

# quality-колонки ──┐
#                   ├── OneHotEncoder
# остальные ────────┘

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
        numerical_steps.append(("scaler", StandardScaler()))

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

    ordinal_steps = [
        (
            "imputer",
            SimpleImputer(
                stategy=(
                    cfg_preprocessing
                    .categorical_imputation
                    .strategy
                )
            )
        ),
        (
            "encoder",
            OrdinalEncoder(
                categories=[
                    list(QUALITY_ORDER)
                    for _ in (ORDINAL_QUALITY_COLUMNS)
                ],
                handle_unknown=("use_encoded_value"),
                unknown_value=-1
            )
        )
    ]

    if cfg_preprocessing.scaling.enabled:
        ordinal_steps.append(
            (
                "scaler",
                StandardScaler()
            )
        )

    ordinal_pipeline = Pipeline(steps=ordinal_steps)

    transformers=[
        (
            "numerical",
            numerical_pipeline,
            make_column_selector(dtype_include=np.number)
        )
    ]

    if cfg_preprocessing.ordinal_encoding.ebabled:
        transformers.append(
            "ordinal_quality",
            ordinal_pipeline,
            list(ORDINAL_QUALITY_COLUMNS)
        )
        categorical_selector = select_nominal_columns
    
    else:
        categorical_selector = make_column_selector(
            dtype_include=[
                "object",
                "category"
            ]
        )
    transformers.append(
        (
            "categorical",
            categorical_pipeline,
            categorical_selector
        )
    )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop", # все колонки, которые не попали ни в один selector, удалить.
        verbose_feature_names_out=False, # не добавлять numerical__ categorical__
                                         # к названиями колонок.
    )
