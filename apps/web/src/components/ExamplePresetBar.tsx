import { WandSparkles } from "lucide-react";
import type { ExampleVehicle } from "../lib/api";

type ExamplePresetBarProps = {
  examples: ExampleVehicle[];
  onSelect: (example: ExampleVehicle) => void;
};

export function ExamplePresetBar({ examples, onSelect }: ExamplePresetBarProps) {
  return (
    <div className="preset-bar" aria-label="Example vehicles">
      {examples.map((example) => (
        <button
          key={example.label}
          className="ghost-button"
          type="button"
          onClick={() => onSelect(example)}
          title={`Load ${example.label}`}
        >
          <WandSparkles size={16} aria-hidden="true" />
          <span>{example.label}</span>
        </button>
      ))}
    </div>
  );
}

