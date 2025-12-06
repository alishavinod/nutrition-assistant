# Nutrition Planner Overview

- **Backend:** FastAPI (`src/app/main.py`), computes macros, optional Ollama LLM plan, dummy pricing.
- **Data:** `data/dummy_catalog.csv` placeholder prices; no live product API.
- **Frontend:** `public/index.html` static UI; works on GitHub Pages, calls backend URL you provide.
- **Config:** `config/app.example.env` for optional env vars (OLLAMA).
- **Scripts:** `scripts/run_backend.sh` to launch dev server.
- **Tests:** `tests/` placeholder; add endpoint and integration tests here.

Future work:
- Replace dummy pricing with real product search/pricing.
- Add nutrition lookup/RAG and tighter budget validation.
- CI (lint/test) via GitHub Actions.
