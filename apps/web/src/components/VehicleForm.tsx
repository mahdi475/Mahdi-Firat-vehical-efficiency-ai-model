import { Gauge, RotateCcw } from "lucide-react";
import type { VehiclePayload } from "../lib/api";

type VehicleFormProps = {
  value: VehiclePayload;
  loading: boolean;
  onChange: (value: VehiclePayload) => void;
  onSubmit: () => void;
  onReset: () => void;
};

type NumericField = {
  key: keyof VehiclePayload;
  label: string;
  min: number;
  max: number;
  step: number;
};

const fields: NumericField[] = [
  { key: "cylinders", label: "Cylinders", min: 3, max: 12, step: 1 },
  { key: "displacement", label: "Displacement", min: 1, max: 1000, step: 0.1 },
  { key: "horsepower", label: "Horsepower", min: 1, max: 500, step: 0.1 },
  { key: "weight", label: "Weight", min: 1, max: 10000, step: 1 },
  { key: "acceleration", label: "Acceleration", min: 1, max: 40, step: 0.1 },
  { key: "model_year", label: "Model year", min: 70, max: 82, step: 1 }
];

export function VehicleForm({
  value,
  loading,
  onChange,
  onSubmit,
  onReset
}: VehicleFormProps) {
  function update(key: keyof VehiclePayload, nextValue: number) {
    onChange({ ...value, [key]: nextValue });
  }

  return (
    <form
      className="vehicle-form"
      onSubmit={(event) => {
        event.preventDefault();
        onSubmit();
      }}
    >
      <div className="field-grid">
        {fields.map((field) => (
          <label className="field" key={field.key}>
            <span>{field.label}</span>
            <input
              type="number"
              min={field.min}
              max={field.max}
              step={field.step}
              value={value[field.key]}
              onChange={(event) => update(field.key, Number(event.target.value))}
            />
          </label>
        ))}

        <label className="field">
          <span>Origin</span>
          <select
            value={value.origin}
            onChange={(event) => update("origin", Number(event.target.value))}
          >
            <option value={1}>USA</option>
            <option value={2}>Europe</option>
            <option value={3}>Japan</option>
          </select>
        </label>
      </div>

      <div className="form-actions">
        <button className="primary-button" type="submit" disabled={loading} title="Predict MPG">
          <Gauge size={18} aria-hidden="true" />
          <span>{loading ? "Predicting" : "Predict"}</span>
        </button>
        <button className="secondary-button" type="button" onClick={onReset} title="Reset form">
          <RotateCcw size={18} aria-hidden="true" />
          <span>Reset</span>
        </button>
      </div>
    </form>
  );
}

