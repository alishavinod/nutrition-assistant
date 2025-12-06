#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uvicorn src.app.main:app --reload --port 8000
