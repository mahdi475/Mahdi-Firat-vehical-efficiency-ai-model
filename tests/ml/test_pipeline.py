from __future__ import annotations

import pandas as pd
from fuel_efficiency.models.baselines import build_model_pipeline


def test_pipeline_fits_and_predicts_with_missing_horsepower():
    x = pd.DataFrame(
        [
            {
                "cylinders": 4,
                "displacement": 100.0,
                "horsepower": 75.0,
                "weight": 2200.0,
                "acceleration": 15.0,
                "model_year": 76,
                "origin": 3,
            },
            {
                "cylinders": 8,
                "displacement": 350.0,
                "horsepower": None,
                "weight": 4200.0,
                "acceleration": 12.0,
                "model_year": 71,
                "origin": 1,
            },
            {
                "cylinders": 6,
                "displacement": 200.0,
                "horsepower": 90.0,
                "weight": 2700.0,
                "acceleration": 15.5,
                "model_year": 75,
                "origin": 1,
            },
        ]
    )
    y = pd.Series([31.0, 14.0, 22.0])

    model = build_model_pipeline(
        "decision_tree",
        ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year"],
        ["origin"],
        random_state=42,
    )
    model.fit(x, y)

    prediction = model.predict(x.iloc[[0]])

    assert prediction.shape == (1,)
