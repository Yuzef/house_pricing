import numpy as np

from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor

from omegaconf import OmegaConf

from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    Ridge,
    LinearRegression
)
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    VotingRegressor,
)

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

def get_model_from_cfg(model_cfg, cat_features=None):
    model_type = str(model_cfg.type)

    model_params = OmegaConf.to_container(
        model_cfg.params,
        resolve=True,
        throw_on_missing=True
    )

    if model_type == "voting_regressor":
        estimators = [
            (
                str(estimator_cfg.name),
                get_model_from_cfg(
                    estimator_cfg,
                    cat_features=cat_features,
                ),
            )
            for estimator_cfg in model_cfg.estimators
        ]

        return VotingRegressor(
            estimators=estimators,
            **model_params,
        )

    if model_type == "linear_regression":
        experiment_model = LinearRegression(
            **model_params
        )

    elif model_type == "ridge":
        experiment_model = Ridge(
            **model_params,
        )

    elif model_type == "lasso":
        experiment_model = Lasso(
            **model_params,
        )

    elif model_type == "elastic_net":
        experiment_model = ElasticNet(
            **model_params,
        )
    
    elif model_type == "knn":
        experiment_model = KNeighborsRegressor(
            **model_params
        )
    
    elif model_type == "decision_tree":
        experiment_model = DecisionTreeRegressor(
            **model_params
        )

    elif model_type == "random_forest":
        experiment_model = RandomForestRegressor(
            **model_params
        )
    
    elif model_type == "catboost":
        catboost_params = dict(model_params)

        if cat_features is not None:
            catboost_params["cat_features"] = tuple(cat_features)

        experiment_model = CatBoostRegressor(
            **catboost_params
        )

    elif model_type == "lightgbm":
        experiment_model = LGBMRegressor(
            **model_params
        )

    elif model_type == "xgboost":
        experiment_model = XGBRegressor(
            **model_params
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    

    return TransformedTargetRegressor(
        regressor=experiment_model,
        func=np.log1p,
        inverse_func=np.expm1
    )

