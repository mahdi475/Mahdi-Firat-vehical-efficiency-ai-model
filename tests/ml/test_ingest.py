from __future__ import annotations

import pandas as pd
from fuel_efficiency.data.ingest import clean_auto_mpg
from fuel_efficiency.data.validate import validate_auto_mpg


def test_clean_auto_mpg_converts_question_mark_to_missing():
    raw = pd.DataFrame(
        [
            {
                "mpg": "18.0",
                "cylinders": "8",
                "displacement": "307.0",
                "horsepower": "?",
                "weight": "3504.",
                "acceleration": "12.0",
                "model_year": "70",
                "origin": "1",
                "car_name": "chevrolet chevelle malibu",
            }
        ]
    )

    cleaned = clean_auto_mpg(raw)

    assert cleaned.loc[0, "horsepower"] != cleaned.loc[0, "horsepower"]
    assert cleaned.loc[0, "mpg"] == 18.0


def test_validate_auto_mpg_reports_missing_horsepower_without_failing():
    df = pd.DataFrame(
        [
            {
                "mpg": 18.0,
                "cylinders": 8,
                "displacement": 307.0,
                "horsepower": None,
                "weight": 3504.0,
                "acceleration": 12.0,
                "model_year": 70,
                "origin": 1,
                "car_name": "chevrolet chevelle malibu",
            }
        ]
    )

    result = validate_auto_mpg(df)

    assert result["valid"] is True
    assert result["missing_values"]["horsepower"] == 1
