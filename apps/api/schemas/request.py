from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    cylinders: int = Field(ge=3, le=12, examples=[4])
    displacement: float = Field(gt=0, le=1000, examples=[140.0])
    horsepower: float = Field(gt=0, le=500, examples=[90.0])
    weight: float = Field(gt=0, le=10000, examples=[2264.0])
    acceleration: float = Field(gt=0, le=40, examples=[15.5])
    model_year: int = Field(ge=70, le=82, examples=[71])
    origin: int = Field(ge=1, le=3, examples=[2])


class BatchPredictionRequest(BaseModel):
    items: list[PredictionRequest] = Field(min_length=1, max_length=100)
