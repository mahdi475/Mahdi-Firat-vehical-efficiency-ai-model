from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from apps.api.config import settings
from apps.api.routers import health, metadata, predict

app = FastAPI(
    title="TAM400 Vehicle Fuel Efficiency API",
    description="Predicts MPG from UCI Auto MPG-style vehicle attributes.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(metadata.router)
app.include_router(predict.router)

static_dist = Path("apps/api/static/dist")
assets_dir = static_dist / "assets"
if assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")


@app.get("/", include_in_schema=False)
def index():
    index_path = static_dist / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "TAM400 Vehicle Fuel Efficiency API", "docs": "/docs"}
