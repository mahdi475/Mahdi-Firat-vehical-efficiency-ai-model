from __future__ import annotations

from fastapi import APIRouter, HTTPException

from apps.api.schemas.request import BatchPredictionRequest, PredictionRequest
from apps.api.schemas.response import BatchPredictionResponse, PredictionResponse
from apps.api.services.model_loader import get_predictor, load_model_card

router = APIRouter(prefix="/api", tags=["prediction"])


@router.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    try:
        predictor = get_predictor()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    predicted_mpg = predictor.predict(payload)
    card = load_model_card()
    return PredictionResponse(
        predicted_mpg=round(predicted_mpg, 2),
        model_version=str(card.get("model_version", "local")),
        input_validated=True,
        notes=card.get(
            "notes",
            ["Historical Auto MPG model", "Use only for vehicles similar to training data"],
        ),
    )


@router.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(payload: BatchPredictionRequest) -> BatchPredictionResponse:
    try:
        predictor = get_predictor()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    card = load_model_card()
    predictions = [
        PredictionResponse(
            predicted_mpg=round(predictor.predict(item), 2),
            model_version=str(card.get("model_version", "local")),
            input_validated=True,
            notes=card.get("notes", []),
        )
        for item in payload.items
    ]
    return BatchPredictionResponse(predictions=predictions)
