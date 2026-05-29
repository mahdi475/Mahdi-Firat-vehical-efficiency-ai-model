from __future__ import annotations

import pandas as pd


def select_model_features(
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    return df[[*numeric_features, *categorical_features]].copy()
