from __future__ import annotations

import pandas as pd

from fuel_efficiency.data.ingest import COLUMNS


def validate_auto_mpg(df: pd.DataFrame) -> dict[str, object]:
    issues: list[str] = []

    if list(df.columns) != COLUMNS:
        issues.append("columns_do_not_match_expected_schema")

    if df.empty:
        issues.append("dataset_is_empty")

    if df["mpg"].isna().any():
        issues.append("target_contains_missing_values")

    range_checks = {
        "mpg": (0, 100),
        "cylinders": (1, 16),
        "displacement": (0, 1000),
        "horsepower": (0, 500),
        "weight": (0, 10000),
        "acceleration": (0, 40),
        "model_year": (0, 99),
        "origin": (1, 3),
    }

    for column, (low, high) in range_checks.items():
        series = df[column].dropna()
        if ((series < low) | (series > high)).any():
            issues.append(f"{column}_outside_expected_range")

    return {
        "row_count": int(len(df)),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": {key: int(value) for key, value in df.isna().sum().to_dict().items()},
        "issues": issues,
        "valid": not issues,
    }
