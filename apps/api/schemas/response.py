from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionUnits(BaseModel):
    predicted_mpg: str = "miles_per_gallon"


class PredictionResponse(BaseModel):
    predicted_mpg: float
    model_version: str
    input_validated: bool
    units: PredictionUnits = Field(default_factory=PredictionUnits)
    notes: list[str] = Field(default_factory=list)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
