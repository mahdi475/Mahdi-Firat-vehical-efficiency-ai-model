from __future__ import annotations

import logging
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

LOGGER = logging.getLogger(__name__)

UCI_AUTO_MPG_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
)

COLUMNS = [
    "mpg",
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model_year",
    "origin",
    "car_name",
]

FEATURE_COLUMNS = [
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model_year",
    "origin",
]

NUMERIC_COLUMNS = [
    "mpg",
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model_year",
    "origin",
]


def ensure_raw_data(raw_path: str | Path, download_if_missing: bool = True) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path

    if not download_if_missing:
        raise FileNotFoundError(f"Missing raw dataset at {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading UCI Auto MPG dataset to %s", path)
    urlretrieve(UCI_AUTO_MPG_URL, path)
    return path


def load_auto_mpg(raw_path: str | Path, download_if_missing: bool = True) -> pd.DataFrame:
    path = ensure_raw_data(raw_path, download_if_missing=download_if_missing)

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = parse_uci_auto_mpg_text(path)

    return clean_auto_mpg(df)


def parse_uci_auto_mpg_text(path: str | Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            parts = stripped.split(maxsplit=8)
            if len(parts) != 9:
                raise ValueError(f"Malformed Auto MPG row at line {line_number}: {stripped}")

            rows.append(dict(zip(COLUMNS, [*parts[:8], parts[8].strip('"')], strict=True)))

    return pd.DataFrame(rows, columns=COLUMNS)


def clean_auto_mpg(df: pd.DataFrame) -> pd.DataFrame:
    missing_columns = set(COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing_columns)}")

    cleaned = df[COLUMNS].copy()
    for column in NUMERIC_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned["car_name"] = cleaned["car_name"].astype(str).str.strip()
    return cleaned


def save_interim(df: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
