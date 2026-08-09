import numpy as np

from omegaconf import OmegaConf

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor

def get_model_from_cfg(model_cfg):
    model_type = str(model_cfg.type)

    model_params = OmegaConf.to_container(
        model_cfg.params,
        resolve=True,
        throw_on_missing=True
    )

    if model_type == "random_forest":
        experiment_model = RandomForestRegressor(
            **model_params
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    

    return TransformedTargetRegressor(
        regressor=experiment_model,
        fung=np.log1p,
        inverse_func=np.expm1
    )

