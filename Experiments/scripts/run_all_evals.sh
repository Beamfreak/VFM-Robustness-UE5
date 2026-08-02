#!/usr/bin/env bash
# Simple wrapper to run the Python runner with unbuffered output.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# Run as a module from the repository root so `scripts` is importable
cd "$REPO_ROOT"
python3 -u -m scripts.run_all_evals "$@"
