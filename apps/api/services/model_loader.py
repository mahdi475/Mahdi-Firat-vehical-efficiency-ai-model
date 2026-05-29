from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fuel_efficiency.data.ingest import FEATURE_COLUMNS

from apps.api.config import settings
from apps.api.schemas.request import PredictionRequest


class Predictor:
    def __init__(self, model: Any):
        self.model = model

    def predict(self, payload: PredictionRequest) -> float:
        frame = pd.DataFrame([{column: getattr(payload, column) for column in FEATURE_COLUMNS}])
        return float(self.model.predict(frame)[0])


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    model_path = settings.model_path
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. Run `npm run train` first."
        )
    return Predictor(joblib.load(model_path))


def load_model_card() -> dict[str, Any]:
    path = Path(settings.model_card_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
