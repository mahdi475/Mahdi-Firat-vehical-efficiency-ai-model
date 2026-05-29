import { test, expect } from "@playwright/test";

test("demo page renders the prediction form", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: /vehicle fuel efficiency/i })).toBeVisible();
  await expect(page.getByLabel(/cylinders/i)).toBeVisible();
});

