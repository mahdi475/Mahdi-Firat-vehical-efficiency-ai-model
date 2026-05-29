export type VehiclePayload = {
  cylinders: number;
  displacement: number;
  horsepower: number;
  weight: number;
  acceleration: number;
  model_year: number;
  origin: number;
};

export type PredictionResponse = {
  predicted_mpg: number;
  model_version: string;
  input_validated: boolean;
  units: { predicted_mpg: string };
  notes: string[];
};

export type MetadataResponse = {
  model_version: string;
  final_model: string;
  features: string[];
  metrics: Record<string, number>;
  notes: string[];
};

export type ExampleVehicle = {
  label: string;
  payload: VehiclePayload;
};

const API_BASE =
  import.meta.env.VITE_API_BASE_URL ??
  (window.location.port === "5173" ? "http://localhost:8000" : "");

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.detail ?? `Request failed with ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export async function predictVehicle(payload: VehiclePayload): Promise<PredictionResponse> {
  const response = await fetch(`${API_BASE}/api/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  return parseResponse<PredictionResponse>(response);
}

export async function fetchMetadata(): Promise<MetadataResponse> {
  const response = await fetch(`${API_BASE}/api/metadata`);
  return parseResponse<MetadataResponse>(response);
}

export async function fetchExamples(): Promise<ExampleVehicle[]> {
  const response = await fetch(`${API_BASE}/api/examples`);
  return parseResponse<ExampleVehicle[]>(response);
}
