from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from fuel_efficiency.data.ingest import FEATURE_COLUMNS

SAMPLE_VEHICLE = {
    "cylinders": 4,
    "displacement": 140.0,
    "horsepower": 90.0,
    "weight": 2264.0,
    "acceleration": 15.5,
    "model_year": 71,
    "origin": 2,
}


def predict_one(model_path: str | Path, payload: dict[str, float | int]) -> float:
    model = joblib.load(model_path)
    frame = pd.DataFrame([{column: payload[column] for column in FEATURE_COLUMNS}])
    return float(model.predict(frame)[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/models/model.joblib")
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()

    if not args.sample:
        raise SystemExit("Only --sample is implemented for CLI prediction.")

    prediction = predict_one(args.model, SAMPLE_VEHICLE)
    print(json.dumps({"input": SAMPLE_VEHICLE, "predicted_mpg": round(prediction, 2)}, indent=2))


if __name__ == "__main__":
    main()
