import { useEffect, useMemo, useState } from "react";
import { Github, Server } from "lucide-react";
import { ExamplePresetBar } from "./components/ExamplePresetBar";
import { PredictionCard } from "./components/PredictionCard";
import { ValidationBanner } from "./components/ValidationBanner";
import { VehicleForm } from "./components/VehicleForm";
import {
  type ExampleVehicle,
  type MetadataResponse,
  type PredictionResponse,
  type VehiclePayload,
  fetchExamples,
  fetchMetadata,
  predictVehicle
} from "./lib/api";

const initialPayload: VehiclePayload = {
  cylinders: 4,
  displacement: 140,
  horsepower: 90,
  weight: 2264,
  acceleration: 15.5,
  model_year: 71,
  origin: 2
};

function validatePayload(payload: VehiclePayload): string | null {
  const checks: Array<[boolean, string]> = [
    [payload.cylinders >= 3 && payload.cylinders <= 12, "Cylinders must be between 3 and 12."],
    [payload.displacement > 0 && payload.displacement <= 1000, "Displacement is outside range."],
    [payload.horsepower > 0 && payload.horsepower <= 500, "Horsepower is outside range."],
    [payload.weight > 0 && payload.weight <= 10000, "Weight is outside range."],
    [payload.acceleration > 0 && payload.acceleration <= 40, "Acceleration is outside range."],
    [payload.model_year >= 70 && payload.model_year <= 82, "Model year must be 70 to 82."],
    [payload.origin >= 1 && payload.origin <= 3, "Origin must be USA, Europe, or Japan."]
  ];
  return checks.find(([ok]) => !ok)?.[1] ?? null;
}

export default function App() {
  const [payload, setPayload] = useState<VehiclePayload>(initialPayload);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [metadata, setMetadata] = useState<MetadataResponse | null>(null);
  const [examples, setExamples] = useState<ExampleVehicle[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const validationMessage = useMemo(() => validatePayload(payload), [payload]);

  useEffect(() => {
    fetchMetadata().then(setMetadata).catch(() => setMetadata(null));
    fetchExamples().then(setExamples).catch(() => setExamples([]));
  }, []);

  async function submit() {
    const validation = validatePayload(payload);
    if (validation) {
      setError(validation);
      return;
    }

    setLoading(true);
    setError(null);
    try {
      setPrediction(await predictVehicle(payload));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">TAM400</p>
          <h1>Vehicle Fuel Efficiency</h1>
        </div>
        <div className="topbar-links">
          <a href="/docs" title="Open API docs">
            <Server size={18} aria-hidden="true" />
            <span>API</span>
          </a>
          <a href="https://github.com/" title="Open GitHub">
            <Github size={18} aria-hidden="true" />
            <span>GitHub</span>
          </a>
        </div>
      </header>

      <section className="workspace">
        <div className="input-panel">
          <div className="panel-heading">
            <p className="eyebrow">Input</p>
            <h2>Vehicle attributes</h2>
          </div>
          <ExamplePresetBar examples={examples} onSelect={(example) => setPayload(example.payload)} />
          <ValidationBanner message={error ?? validationMessage} />
          <VehicleForm
            value={payload}
            loading={loading}
            onChange={setPayload}
            onSubmit={submit}
            onReset={() => {
              setPayload(initialPayload);
              setPrediction(null);
              setError(null);
            }}
          />
        </div>

        <PredictionCard prediction={prediction} metadata={metadata} />
      </section>

      <footer className="footer">
        <span>Dataset: UCI Auto MPG, CC BY 4.0.</span>
        <span>AI assistance acknowledged in report and slides.</span>
      </footer>
    </main>
  );
}

