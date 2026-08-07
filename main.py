from utils.load_data import load_data_func
from config import config
from pathlib import Path

from utils.preprocessing import build_preprocessor

def main() -> None:

    target_column = config.target.name
    id_column = config.id_column

    train_df, test_df = load_data_func(
        config.paths.train_csv,
        config.paths.test_csv,
    )

    X_train = train_df.drop(columns=[id_column, target_column])
    y_train = train_df[target_column]

    X_test = test_df.drop(columns=id_column)
    test_ids = test_df[id_column].copy()

    preprocessor = build_preprocessor(config.preprocessing)

    base_model = RandomForestRegressor(
        n_estimators=100,
        random_state=config.general.seed
    )

    model = TransformedTargetRegressor(
        regressor=base_model,
        func=np.log1p,
        inverse_func=np.expm1
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model)
        ]
    )




if __name__ == "__main__":
    main()
