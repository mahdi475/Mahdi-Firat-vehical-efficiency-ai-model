import { cpSync, existsSync, rmSync } from "node:fs";
import path from "node:path";

const root = process.cwd();
const source = path.join(root, "apps", "web", "dist");
const target = path.join(root, "apps", "api", "static", "dist");

if (!existsSync(source)) {
  console.error("Missing apps/web/dist. Run npm run build:web first.");
  process.exit(1);
}

rmSync(target, { recursive: true, force: true });
cpSync(source, target, { recursive: true });
console.log(`Copied ${source} to ${target}`);

