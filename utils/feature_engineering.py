import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

def add_age_features(X: pd.DataFrame) -> pd.DataFrame:

    X = X.copy()

    X["HouseAge"] = X["YrSold"] - X["YearBuilt"]
    X["YearsSinceRemodel"] = X["YrSold"] - X["YearRemodAdd"]
    X["GarageAge"]= X["YrSold"] - X["GarageYrBlt"]
    X["IsRemodeled"]= (X["YearRemodAdd"] != X["YearBuilt"]).astype("int8")

    return X

def add_area_features(X: pd.DataFrame) -> pd.DataFrame:

    X = X.copy()

    X["TotalSF"] = X["GrLivArea"] + X["TotalBsmtSF"]
    X["TotalFinishedSF"] = X["GrLivArea"] + X["BsmtFinSF1"] + X["BsmtFinSF2"]

    porch_columns = [
        "OpenPorchSF",
        "EnclosedPorch",
        "3SsnPorch",
        "ScreenPorch",
        "WoodDeckSF"
    ]

    X["TotalPorchSF"] = X[
        porch_columns
        ].sum(axis=1, min_count=len(porch_columns))
        # min_count=len(porch_columns) - Все 5 значений должны быть.
        # Считай TotalPorchSF только тогда,
        # когда известны значения всех компонентов

    return X

def add_bathroom_features(X: pd.DataFrame) -> pd.DataFrame:

    X = X.copy()

    X["TotalBathrooms"] = (
        X["FullBath"]
        + 0.5 * X["HalfBath"]
        + X["BsmtFullBath"]
        + 0.5 * X["BsmtHalfBath"]
    )

    return X

def add_quality_area_interaction(X: pd.DataFrame) -> pd.DataFrame:

    X = X.copy()

    X["OverallQual_GrLivArea"] = (
        X["OverallQual"] * X["GrLivArea"]
    )

    return X

def build_feature_engineer(config):
    steps = []

    if config.age_features.enabled:
        steps.append(
            (
                "age_features",
                FunctionTransformer(
                    add_age_features,
                    validate=False
                )
            )
        )

    if config.area_features.enabled:
        steps.append(
            (
                "area_features",
                FunctionTransformer(
                    add_area_features,
                    validate=False,
                ),
            )
        )

    if config.bathroom_features.enabled:
            steps.append(
                (
                    "bathroom_features",
                    FunctionTransformer(
                        add_bathroom_features,
                        validate=False,
                    ),
                )
            )

    if config.quality_area_interaction.enabled:
        steps.append(
            (
                "quality_area_interaction",
                FunctionTransformer(
                    add_quality_area_interaction,
                    validate=False,
                ),
            )
        )

    if not steps:
        return FunctionTransformer(validate=False)

    return Pipeline(steps=steps)

