import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";

const root = process.cwd();
const [scriptPath, ...scriptArgs] = process.argv.slice(2);

if (!scriptPath) {
  console.error("Usage: node scripts/run-python-file.mjs <script.py> [args...]");
  process.exit(2);
}

const bundledPython = path.join(
  process.env.USERPROFILE ?? "",
  ".cache",
  "codex-runtimes",
  "codex-primary-runtime",
  "dependencies",
  "python",
  "python.exe"
);

const candidates = [
  path.join(root, ".venv", "Scripts", "python.exe"),
  path.join(root, ".venv", "bin", "python"),
  bundledPython,
  "python",
  "py"
];

const python = candidates.find((candidate) => {
  return candidate === "python" || candidate === "py" || existsSync(candidate);
});

const env = {
  ...process.env,
  PYTHONPATH: [path.join(root, "ml", "src"), root, process.env.PYTHONPATH]
    .filter(Boolean)
    .join(path.delimiter)
};

const child = spawn(python, [scriptPath, ...scriptArgs], { stdio: "inherit", env });
child.on("exit", (code) => process.exit(code ?? 1));

