# Nutrition Planner Demo

Showcase-ready repo with a structured backend/frontend split. FastAPI powers the API; a static UI works on GitHub Pages and calls your backend. Uses a dummy pricing catalog (no real product API) and an optional local LLM via Ollama.

## Repo Layout
- `src/app/` — FastAPI app.
- `public/` — Static demo UI (GitHub Pages friendly).
- `data/` — Dummy pricing CSV (placeholder).
- `config/` — Example env (`app.example.env`).
- `scripts/` — Dev helper (`run_backend.sh`).
- `tests/` — Placeholder tests.
- `docs/` — Overview/notes.
- `.github/` — Contributing and code of conduct.
- `requirements.txt` — Backend deps.

## Project Layout
- `backend/` — FastAPI app (`main.py`), dependencies (`requirements.txt`), dummy data (`data/dummy_catalog.csv`).
- `frontend/` — Static UI (`index.html`) that talks to the backend (default `http://localhost:8000`).
- `README.md` — You are here.

## Backend (src/app)

### Requirements
- Python 3.9+
- `pip install -r requirements.txt`
- Optional: [Ollama](https://ollama.com/download) with a pulled model (e.g., `ollama pull llama3`) and env `OLLAMA_MODEL=llama3`.

### Run
```bash
cd src
uvicorn app.main:app --reload --port 8000
```

Health check:
```bash
curl http://127.0.0.1:8000/health
```

Generate a plan (LLM optional via `use_llm`):
```bash
curl -X POST http://127.0.0.1:8000/plan \
  -H 'Content-Type: application/json' \
  -d '{
    "profile": {"height_cm":175,"weight_kg":70,"age":30,"sex":"male","activity_level":"moderate"},
    "dietary_preference":"omnivore",
    "budget_amount":100,
    "budget_period":"week",
    "meals_per_day":3,
    "use_llm": false
  }'
```

Notes:
- Cost estimation uses `backend/data/dummy_catalog.csv`; quantities are ignored and prices are placeholders.
- No live product search; replace the dummy catalog with real pricing when available.
- CORS is open to allow the static frontend to call the backend system.

## Frontend (GitHub Pages friendly)

`public/index.html` is a static demo UI. Serve it from GitHub Pages or any static host. It defaults to calling `http://localhost:8000`. If your backend runs elsewhere, change the Backend URL field in the page.

To view locally:
```bash
python -m http.server 3000 --directory public
```
Then open http://localhost:3000 and point the Backend URL to your running backend (e.g., http://localhost:8000).

## Deployment Notes
- Frontend: push `frontend/` to a `gh-pages` branch or any static host.
- Backend: deploy separately (Render/Fly/railway/etc.) and update the Backend URL in the UI.
