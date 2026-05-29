from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_validate

from fuel_efficiency.data.ingest import load_auto_mpg, save_interim
from fuel_efficiency.data.split import make_train_test_split, split_features_target
from fuel_efficiency.data.validate import validate_auto_mpg
from fuel_efficiency.models.baselines import (
    RF_PARAM_GRID,
    TREE_PARAM_GRID,
    build_model_pipeline,
    model_names,
)
from fuel_efficiency.models.evaluate import regression_metrics
from fuel_efficiency.models.interpret import save_permutation_importance
from fuel_efficiency.utils.io import append_jsonl, read_yaml, write_json
from fuel_efficiency.utils.logging import configure_logging
from fuel_efficiency.utils.seed import set_global_seed

SCORING = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
}


def summarize_cv(cv_results: dict[str, Any]) -> dict[str, float]:
    return {
        "cv_mae_mean": float((-cv_results["test_mae"]).mean()),
        "cv_mae_std": float(cv_results["test_mae"].std()),
        "cv_rmse_mean": float((-cv_results["test_rmse"]).mean()),
        "cv_rmse_std": float(cv_results["test_rmse"].std()),
        "cv_r2_mean": float(cv_results["test_r2"].mean()),
        "cv_r2_std": float(cv_results["test_r2"].std()),
        "fit_time_mean": float(cv_results["fit_time"].mean()),
    }


def tune_if_requested(
    model_name: str,
    pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int,
    n_jobs: int,
    tune_random_forest: bool,
):
    if model_name == "decision_tree":
        return GridSearchCV(
            pipeline,
            TREE_PARAM_GRID,
            cv=cv_folds,
            scoring="neg_root_mean_squared_error",
            n_jobs=n_jobs,
        )

    if model_name == "random_forest" and tune_random_forest:
        return GridSearchCV(
            pipeline,
            RF_PARAM_GRID,
            cv=cv_folds,
            scoring="neg_root_mean_squared_error",
            n_jobs=n_jobs,
        )

    return pipeline


def save_residual_plot(y_test, predictions, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    residuals = y_test - predictions

    plt.figure(figsize=(7, 4.5))
    plt.scatter(predictions, residuals, alpha=0.75, edgecolor="none")
    plt.axhline(0, color="#222222", linewidth=1)
    plt.xlabel("Predicted MPG")
    plt.ylabel("Residual MPG")
    plt.title("Final model residuals")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def train_from_config(config_path: str | Path = "ml/configs/base.yaml") -> dict[str, Any]:
    configure_logging()
    config = read_yaml(config_path)
    paths = config["paths"]
    data_config = config["data"]
    feature_config = config["features"]
    training_config = config["training"]

    random_state = int(data_config["random_state"])
    set_global_seed(random_state)

    df = load_auto_mpg(paths["raw_data"], download_if_missing=data_config["download_if_missing"])
    validation = validate_auto_mpg(df)
    if not validation["valid"]:
        raise ValueError(f"Dataset validation failed: {validation['issues']}")

    save_interim(df, paths["interim_data"])

    feature_columns = [*feature_config["numeric"], *feature_config["categorical"]]
    x, y = split_features_target(df, feature_columns, feature_config["target"])
    x_train, x_test, y_train, y_test = make_train_test_split(
        x,
        y,
        test_size=float(data_config["test_size"]),
        random_state=random_state,
    )

    rows: list[dict[str, Any]] = []
    trained_models: dict[str, Any] = {}

    for model_name in model_names(include_random_forest=True):
        pipeline = build_model_pipeline(
            model_name,
            feature_config["numeric"],
            feature_config["categorical"],
            random_state=random_state,
            n_jobs=int(training_config["n_jobs"]),
        )
        estimator = tune_if_requested(
            model_name,
            pipeline,
            x_train,
            y_train,
            cv_folds=int(training_config["cv_folds"]),
            n_jobs=int(training_config["n_jobs"]),
            tune_random_forest=bool(training_config["tune_random_forest"]),
        )

        best_params = {}
        if isinstance(estimator, GridSearchCV):
            estimator.fit(x_train, y_train)
            best_params = estimator.best_params_
            cv_estimator = estimator.best_estimator_
        else:
            cv_estimator = estimator

        cv_results = cross_validate(
            cv_estimator,
            x_train,
            y_train,
            cv=int(training_config["cv_folds"]),
            scoring=SCORING,
            return_train_score=True,
            n_jobs=1,
        )

        best_estimator = cv_estimator.fit(x_train, y_train)
        predictions = best_estimator.predict(x_test)
        test_metrics = {
            f"test_{name}": value for name, value in regression_metrics(y_test, predictions).items()
        }

        row = {
            "model": model_name,
            **summarize_cv(cv_results),
            **test_metrics,
            "best_params": best_params,
        }
        rows.append(row)
        trained_models[model_name] = best_estimator

    comparison = pd.DataFrame(rows).sort_values("test_rmse")
    Path(paths["comparison"]).parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(paths["comparison"], index=False)

    final_model_name = str(training_config["final_model"])
    final_model = trained_models[final_model_name]
    final_predictions = final_model.predict(x_test)
    final_metrics = regression_metrics(y_test, final_predictions)

    model_path = Path(paths["model"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, model_path)

    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    model_version = f"{config['project']['model_version_prefix']}-{timestamp}"
    model_card = {
        "model_version": model_version,
        "trained_at_utc": timestamp,
        "dataset": "UCI Auto MPG",
        "license": "CC BY 4.0",
        "final_model": final_model_name,
        "features": feature_columns,
        "target": feature_config["target"],
        "validation": validation,
        "metrics": {f"test_{key}": value for key, value in final_metrics.items()},
        "best_params": comparison.loc[comparison["model"] == final_model_name, "best_params"].iloc[
            0
        ],
        "notes": [
            "Historical Auto MPG model",
            "Use only for vehicles similar to the training data",
        ],
    }

    write_json(paths["model_card"], model_card)
    write_json(paths["metrics"], model_card["metrics"])
    append_jsonl(paths["experiments"], model_card)

    figures_dir = Path(paths["figures_dir"])
    save_residual_plot(y_test, final_predictions, figures_dir / "residuals.png")
    save_permutation_importance(
        final_model,
        x_test,
        y_test,
        figures_dir / "permutation_importance.csv",
        random_state=random_state,
    )

    return {"model_card": model_card, "comparison": comparison.to_dict(orient="records")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ml/configs/base.yaml")
    args = parser.parse_args()

    result = train_from_config(args.config)
    print(pd.DataFrame(result["comparison"]).to_string(index=False))


if __name__ == "__main__":
    main()
