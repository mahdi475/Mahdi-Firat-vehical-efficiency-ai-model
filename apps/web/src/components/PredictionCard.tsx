import { Activity, BadgeCheck, Database } from "lucide-react";
import type { MetadataResponse, PredictionResponse } from "../lib/api";

type PredictionCardProps = {
  prediction: PredictionResponse | null;
  metadata: MetadataResponse | null;
};

function formatMetric(value: number | undefined) {
  if (typeof value !== "number") {
    return "-";
  }
  return value.toFixed(2);
}

export function PredictionCard({ prediction, metadata }: PredictionCardProps) {
  const mpg = prediction?.predicted_mpg;
  const band = mpg === undefined ? "Waiting" : mpg >= 30 ? "High" : mpg >= 20 ? "Medium" : "Low";

  return (
    <section className="prediction-panel" aria-label="Prediction result">
      <div className="prediction-header">
        <div>
          <p className="eyebrow">Prediction</p>
          <h2>{mpg === undefined ? "--.-" : mpg.toFixed(1)} MPG</h2>
        </div>
        <div className={`band band-${band.toLowerCase()}`}>{band}</div>
      </div>

      <div className="metric-strip">
        <div>
          <Activity size={18} aria-hidden="true" />
          <span>MAE</span>
          <strong>{formatMetric(metadata?.metrics?.test_mae)}</strong>
        </div>
        <div>
          <BadgeCheck size={18} aria-hidden="true" />
          <span>R²</span>
          <strong>{formatMetric(metadata?.metrics?.test_r2)}</strong>
        </div>
        <div>
          <Database size={18} aria-hidden="true" />
          <span>Model</span>
          <strong>{metadata?.final_model ?? "untrained"}</strong>
        </div>
      </div>

      <div className="model-notes">
        <p>Version: {prediction?.model_version ?? metadata?.model_version ?? "untrained"}</p>
        {(prediction?.notes ?? metadata?.notes ?? []).slice(0, 2).map((note) => (
          <p key={note}>{note}</p>
        ))}
      </div>
    </section>
  );
}

