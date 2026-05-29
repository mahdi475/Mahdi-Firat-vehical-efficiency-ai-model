from __future__ import annotations

from fastapi import APIRouter

from apps.api.services.model_loader import load_model_card

router = APIRouter(prefix="/api", tags=["metadata"])


EXAMPLES = [
    {
        "label": "Compact efficient",
        "payload": {
            "cylinders": 4,
            "displacement": 97.0,
            "horsepower": 75.0,
            "weight": 2171.0,
            "acceleration": 16.0,
            "model_year": 75,
            "origin": 3,
        },
    },
    {
        "label": "Mid-size balanced",
        "payload": {
            "cylinders": 6,
            "displacement": 199.0,
            "horsepower": 90.0,
            "weight": 2648.0,
            "acceleration": 15.0,
            "model_year": 70,
            "origin": 1,
        },
    },
    {
        "label": "Heavy classic",
        "payload": {
            "cylinders": 8,
            "displacement": 350.0,
            "horsepower": 165.0,
            "weight": 4209.0,
            "acceleration": 12.0,
            "model_year": 71,
            "origin": 1,
        },
    },
]


@router.get("/metadata")
def metadata() -> dict[str, object]:
    card = load_model_card()
    return {
        "model_version": card.get("model_version", "untrained"),
        "final_model": card.get("final_model", "untrained"),
        "features": card.get(
            "features",
            [
                "cylinders",
                "displacement",
                "horsepower",
                "weight",
                "acceleration",
                "model_year",
                "origin",
            ],
        ),
        "metrics": card.get("metrics", {}),
        "notes": card.get(
            "notes",
            ["Train the model with npm run train before using production predictions."],
        ),
    }


@router.get("/examples")
def examples() -> list[dict[str, object]]:
    return EXAMPLES
