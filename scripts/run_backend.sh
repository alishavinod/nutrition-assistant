#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Auto-load local environment overrides if present.
if [ -f ".env" ]; then
  set -a
  # shellcheck source=/dev/null
  . ".env"
  set +a
fi

uvicorn src.app.main:app --reload --port 8000
