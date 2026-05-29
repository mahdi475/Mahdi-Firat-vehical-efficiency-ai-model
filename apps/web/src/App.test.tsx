import "@testing-library/jest-dom/vitest";
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import App from "./App";

vi.stubGlobal(
  "fetch",
  vi.fn((url: string) => {
    if (url.endsWith("/api/metadata")) {
      return Promise.resolve(
        new Response(
          JSON.stringify({
            model_version: "test",
            final_model: "random_forest",
            features: [],
            metrics: { test_mae: 2.1, test_r2: 0.82 },
            notes: []
          }),
          { status: 200 }
        )
      );
    }

    return Promise.resolve(new Response(JSON.stringify([]), { status: 200 }));
  })
);

describe("App", () => {
  it("renders the demo shell", async () => {
    render(<App />);

    expect(screen.getByRole("heading", { name: /vehicle fuel efficiency/i })).toBeInTheDocument();
    expect(screen.getByLabelText(/cylinders/i)).toBeInTheDocument();
    expect(await screen.findByText(/random_forest/i)).toBeInTheDocument();
  });
});
