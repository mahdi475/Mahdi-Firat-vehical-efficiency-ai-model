from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from fuel_efficiency.data.ingest import load_auto_mpg
from fuel_efficiency.data.split import make_train_test_split, split_features_target
from fuel_efficiency.utils.io import read_yaml


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_saved_model(config_path: str | Path) -> dict[str, float]:
    config = read_yaml(config_path)
    paths = config["paths"]
    data_config = config["data"]
    feature_config = config["features"]

    df = load_auto_mpg(paths["raw_data"], download_if_missing=data_config["download_if_missing"])
    x, y = split_features_target(
        df,
        [*feature_config["numeric"], *feature_config["categorical"]],
        feature_config["target"],
    )
    _, x_test, _, y_test = make_train_test_split(
        x,
        y,
        test_size=data_config["test_size"],
        random_state=data_config["random_state"],
    )

    model = joblib.load(paths["model"])
    predictions = model.predict(x_test)
    return regression_metrics(y_test, predictions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ml/configs/base.yaml")
    args = parser.parse_args()

    metrics = evaluate_saved_model(args.config)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
