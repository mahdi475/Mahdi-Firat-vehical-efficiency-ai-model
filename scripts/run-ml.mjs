import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";

const root = process.cwd();
const [moduleName, ...moduleArgs] = process.argv.slice(2);

if (!moduleName) {
  console.error("Usage: node scripts/run-ml.mjs <python.module> [args...]");
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

const child = spawn(python, ["-m", moduleName, ...moduleArgs], { stdio: "inherit", env });
child.on("exit", (code) => process.exit(code ?? 1));
