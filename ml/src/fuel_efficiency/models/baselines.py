from __future__ import annotations

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from fuel_efficiency.features.preprocess import build_preprocessor


def build_model_pipeline(
    model_name: str,
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int,
    n_jobs: int | None = None,
) -> Pipeline:
    scale_numeric = model_name == "linear_regression"
    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
        scale_numeric=scale_numeric,
    )

    if model_name == "dummy":
        estimator = DummyRegressor(strategy="mean")
    elif model_name == "linear_regression":
        estimator = LinearRegression()
    elif model_name == "decision_tree":
        estimator = DecisionTreeRegressor(
            max_depth=5, min_samples_leaf=2, random_state=random_state
        )
    elif model_name == "random_forest":
        estimator = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=1,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return Pipeline(steps=[("preprocess", preprocessor), ("estimator", estimator)])


def model_names(include_random_forest: bool = True) -> list[str]:
    names = ["dummy", "linear_regression", "decision_tree"]
    if include_random_forest:
        names.append("random_forest")
    return names


TREE_PARAM_GRID = {
    "estimator__max_depth": [3, 5, 8, None],
    "estimator__min_samples_split": [2, 5, 10],
    "estimator__min_samples_leaf": [1, 2, 4],
    "estimator__ccp_alpha": [0.0, 0.001, 0.01],
}

RF_PARAM_GRID = {
    "estimator__n_estimators": [100, 300],
    "estimator__max_depth": [None, 5, 10],
    "estimator__min_samples_leaf": [1, 2, 4],
    "estimator__max_features": [1.0, "sqrt", 0.5],
}
