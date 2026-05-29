# Architecture

The repository keeps the machine learning code, API, and web demo in one project so the
course submission is easy to run and explain.

```mermaid
flowchart LR
  raw["UCI Auto MPG raw data"] --> train["Train / evaluate CLI"]
  train --> artifact["model.joblib + model_card.json"]
  artifact --> api["FastAPI /api/predict"]
  api --> web["React + Vite demo"]
```

The key contract is the serialized scikit-learn pipeline. Preprocessing and the final
regressor are saved together, which keeps API inference aligned with training.

