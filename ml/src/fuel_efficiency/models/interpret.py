from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.inspection import permutation_importance


def save_permutation_importance(
    model,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: str | Path,
    random_state: int,
) -> pd.DataFrame:
    result = permutation_importance(
        model,
        x_test,
        y_test,
        n_repeats=20,
        random_state=random_state,
        scoring="neg_root_mean_squared_error",
    )

    importance = pd.DataFrame(
        {
            "feature": x_test.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(path, index=False)
    return importance
