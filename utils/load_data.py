from pathlib import Path

import pandas as pd

# В этих колонках "NA" —
# полноценная категория, обозначающая отсутствие объекта.
AMES_NA_CATEGORY_COLUMNS = frozenset(
    {
        "Alley",
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2",
        "FireplaceQu",
        "GarageType",
        "GarageFinish",
        "GarageQual",
        "GarageCond",
        "PoolQC",
        "Fence",
        "MiscFeature",
    }
)

def read_project_csv(path) -> pd.DataFrame:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"CSV file not found"
            f" {path}."
            )
    
    columns = pd.read_csv(
        path,
        nrows=0,
    ).columns

    na_values = {
        column: (
            [""] # вот такие значения считать пропусками.
            if column in AMES_NA_CATEGORY_COLUMNS
            else ["", "NA"] # а тут такие значения считать пропусками.
        )
        for column in columns
    }

    return pd.read_csv(path, keep_default_na=False, na_values=na_values)

def load_data_func(
    train_path,
    test_path,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    train_df = read_project_csv(Path(train_path))
    test_df = read_project_csv(Path(test_path))

    return train_df, test_df