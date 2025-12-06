```mermaid
flowchart TD
  subgraph Client
    UI[Static UI\npublic/index.html]
    APIcaller[API Client\n(curl/postman)]
  end

  subgraph Backend[FastAPI\nsrc/app/main.py]
    health[GET /health]
    plan[POST /plan]
    macro[Macro Calc\n(Mifflin-St Jeor)]
    llm[Optional LLM\n(Ollama)]
    stub[Stub Meal Plan]
    cost[Dummy Cost Estimator]
    data[(data/dummy_catalog.csv)]
  end

  UI -->|HTTP| plan
  APIcaller -->|HTTP| plan
  plan --> macro
  plan --> llm
  plan --> stub
  plan --> cost
  cost --> data
  plan --> health

  subgraph Config
    env[config/app.example.env\n(OLLAMA_MODEL, OLLAMA_URL)]
  end
  env --> llm

  subgraph FrontendHosting
    gh[GitHub Pages / static host]
  end
  UI <-->|fetch| plan

  subgraph Deployment
    uvicorn[uvicorn src.app.main:app\n(port 8001)]
  end
  Backend --> uvicorn
```
