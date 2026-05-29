from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model_path: Path = Path(os.getenv("MODEL_PATH", "artifacts/models/model.joblib"))
    model_card_path: Path = Path(os.getenv("MODEL_CARD_PATH", "artifacts/models/model_card.json"))
    cors_origins: tuple[str, ...] = (
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    )


settings = Settings()
