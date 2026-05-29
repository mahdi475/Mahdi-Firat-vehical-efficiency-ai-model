# TAM400 Vehicle Fuel Efficiency

Predict vehicle fuel efficiency (MPG) from simple tabular vehicle attributes using the
UCI Auto MPG dataset, scikit-learn, FastAPI, and a small React/Vite demo.

## Project Shape

- `ml/src/fuel_efficiency` contains reusable data, preprocessing, training, evaluation, and inference code.
- `apps/api` contains the FastAPI prediction service.
- `apps/web` contains the React + Vite demo UI.
- `data/raw` stores the source dataset.
- `artifacts` stores generated metrics, figures, and model files.
- `docs` stores report and presentation support material.

## Dataset

Primary dataset: UCI Auto MPG.

- Task: regression.
- Target: `mpg`.
- Features: `cylinders`, `displacement`, `horsepower`, `weight`, `acceleration`, `model_year`, `origin`.
- Dropped from the main model: `car_name`.
- License: CC BY 4.0.
- DOI: `10.24432/C5859H`.

The training command downloads `auto-mpg.data` into `data/raw` if the file is missing.
For a presentation or final submission, keep the downloaded raw file in the repository
with this attribution.

## Quickstart

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

npm install
npm --prefix apps/web install

npm run train
npm run dev
```

Open:

- Web UI: `http://localhost:5173`
- API docs: `http://localhost:8000/docs`

## Useful Commands

```bash
npm run train
npm run train:lab
npm run evaluate
npm run predict:sample
npm run test:api
npm run test:web
npm run build
npm run docker:up
```

## Model Workflow

The training script compares:

- `DummyRegressor`
- `LinearRegression`
- `DecisionTreeRegressor`
- `RandomForestRegressor`

It saves:

- `artifacts/models/model.joblib`
- `artifacts/models/model_card.json`
- `artifacts/metrics/model_comparison.csv`
- `artifacts/metrics/latest_metrics.json`
- `artifacts/figures/residuals.png`
- `artifacts/figures/permutation_importance.csv`

The saved artifact is one scikit-learn pipeline containing preprocessing and the final
estimator, so API inference uses the same transformations as training.

For presentation and oral explanation, use
`notebooks/01_vehicle_fuel_efficiency_lab_style.py`. It is written like the course
labs with clear tasks, direct variables, cross-validation, grid search, metrics, and
one final saved model.

The Swedish code walkthrough is in `docs/lab_style_code_explanation_sv.txt`.

## GitHub Citation Template

```text
[5] F. Kaya and M. M. Mosavi, "Predicting Vehicle Fuel Efficiency," GitHub repository, 2026.
[Online]. Available: https://github.com/<org>/<repo>. [Accessed: DD-MMM-YYYY].
```

## AI Acknowledgment Template

```text
Generative AI tools were used for brainstorming, code debugging, documentation
structuring, and language polishing. All final implementation, results, and
explanations were reviewed, verified, and understood by both authors.
```

## Limitations

The Auto MPG dataset is small and historical. Predictions should be presented as a
course demonstration for vehicles similar to the training data, not as a general model
for modern vehicle engineering.
