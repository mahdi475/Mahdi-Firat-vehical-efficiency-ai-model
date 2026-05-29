from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def split_features_target(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    return df[feature_columns].copy(), df[target_column].copy()


def make_train_test_split(
    x: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
):
    return train_test_split(x, y, test_size=test_size, random_state=random_state)
