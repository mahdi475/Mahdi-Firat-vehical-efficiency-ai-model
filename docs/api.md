# API

Start the API:

```bash
npm run dev:api
```

Useful endpoints:

- `GET /api/health`
- `GET /api/metadata`
- `GET /api/examples`
- `POST /api/predict`
- `POST /api/predict/batch`
- `GET /docs`

Example prediction payload:

```json
{
  "cylinders": 4,
  "displacement": 140.0,
  "horsepower": 90.0,
  "weight": 2264.0,
  "acceleration": 15.5,
  "model_year": 71,
  "origin": 2
}
```

