# CampusGPT — Production RAG AI Teaching Assistant

> **Flagship Applied AI / ML Engineering portfolio project** for RGPV university students.  
> Retrieval-Augmented Generation (RAG) system with JWT auth, Redis memory, CrossEncoder reranking, LangGraph agents, and RAGAS evaluation.

[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange)](https://www.trychroma.com/)
[![Gemini](https://img.shields.io/badge/Gemini%202.5%20Flash-4285F4?style=flat&logo=google&logoColor=white)](https://ai.google.dev/)

---

## Overview

CampusGPT answers academic questions grounded in uploaded course materials (PDFs, DOCX, PPTX, TXT). Built for **Data Science, AI/ML, and Computer Science** students at RGPV.

| Capability | Technology |
|---|---|
| LLM | Google Gemini 2.5 Flash |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Reranker | `BAAI/bge-reranker-base` (CrossEncoder) |
| Vector DB | ChromaDB (persistent) |
| Auth | JWT + bcrypt |
| Database | SQLite (SQLAlchemy ORM) |
| Memory | Redis (with in-memory fallback) |
| Agent | LangGraph (Planner → Retriever → Memory → Generator) |
| Evaluation | RAGAS (Faithfulness, Context Precision, Answer Relevancy) |
| Frontend | Streamlit (dark mode, portfolio UI) |
| Deploy | Docker, Render, Streamlit Cloud |

---

## Architecture

```
┌─────────────┐     JWT      ┌──────────────────────────────────────────────┐
│  Streamlit  │ ──────────►  │              FastAPI Backend                 │
│  Frontend   │              │  ┌─────────┐  ┌──────────┐  ┌─────────────┐  │
└─────────────┘              │  │  Auth   │  │ LangGraph│  │  RAGAS Eval │  │
                             │  │ JWT+bcrypt│ │  Agent   │  │  Endpoint   │  │
                             │  └────┬────┘  └────┬─────┘  └─────────────┘  │
                             │       │            │                          │
                             │  ┌────▼────┐  ┌────▼─────────────────────┐   │
                             │  │ SQLite  │  │      RAG Pipeline        │   │
                             │  │  ORM    │  │ Embed → Retrieve(10)     │   │
                             │  └─────────┘  │ → Rerank(3) → Gemini     │   │
                             │               └──────────┬───────────────┘   │
                             │  ┌─────────┐             │                   │
                             │  │  Redis  │◄────────────┘                   │
                             │  │ Memory  │                                 │
                             │  └─────────┘                                 │
                             └──────────────────┬───────────────────────────┘
                                                │
                                    ┌───────────▼───────────┐
                                    │       ChromaDB          │
                                    │   (vector + metadata)   │
                                    └─────────────────────────┘
```

### RAG Pipeline

```
Question → Embedding Search (top 10) → CrossEncoder Rerank (top 3) → Context + Gemini → Answer + Citations
```

---

## Folder Structure

```
├── backend/
│   ├── main.py              # FastAPI app
│   ├── config.py            # Pydantic settings
│   ├── models.py            # API schemas
│   ├── auth/                # JWT, bcrypt, routes
│   ├── database/            # SQLAlchemy ORM + services
│   ├── memory/              # Redis conversation memory
│   ├── ingestion/           # Load → Chunk → Embed → Store
│   ├── vectorstore/         # ChromaDB
│   ├── rag/                 # Engine, reranker, agent, citations
│   └── evaluation/          # RAGAS evaluator
├── frontend/app.py          # Streamlit UI
├── tests/                   # 39+ tests
├── scripts/                 # run_app.py, init_db.py
├── data/                    # uploads, sample notes, SQLite DB
├── docker-compose.yml       # backend + frontend + redis
├── requirements.txt
├── Procfile                 # Render deployment
└── runtime.txt
```

---

## Installation

### Prerequisites
- Python 3.11+
- Redis (optional — falls back to in-memory)
- Google Gemini API key

### Setup

```bash
git clone <your-repo>
cd "AI Teaching Assistant for RGPV Students"

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env — set GEMINI_API_KEY and JWT_SECRET_KEY

python scripts/init_db.py       # Create SQLite tables
```

### Run Locally

```bash
# Terminal 1 — Backend
python scripts/run_app.py backend

# Terminal 2 — Frontend
python scripts/run_app.py frontend

# Optional — Redis
docker run -d -p 6379:6379 redis:7-alpine
```

- **API:** http://localhost:8000/docs
- **UI:** http://localhost:8501

### Docker (all services)

```bash
docker-compose up --build
```

---

## API Documentation

### Auth
| Endpoint | Method | Description |
|---|---|---|
| `/auth/register` | POST | Create account |
| `/auth/login` | POST | Get JWT token |
| `/auth/logout` | POST | Revoke session |
| `/auth/me` | GET | Current user |

### Protected Endpoints (require `Authorization: Bearer <token>`)
| Endpoint | Method | Description |
|---|---|---|
| `/chat` | POST | RAG chat (6 modes) |
| `/ingest` | POST | Upload document |
| `/documents` | GET/DELETE | List/delete docs |
| `/metrics` | GET | Analytics + feedback stats |
| `/feedback` | POST | Submit 👍/👎 feedback |
| `/feedback/stats` | GET | Feedback analytics |
| `/evaluation` | POST | Run RAGAS evaluation |
| `/evaluation/history` | GET | Past evaluation runs |

---

## Deployment

### Backend — Render

1. Push to GitHub
2. Create **Web Service** on [Render](https://render.com)
3. Build: `pip install -r requirements.txt`
4. Start: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
5. Set env vars from `.env.example`
6. Add Redis addon or set `REDIS_ENABLED=false`

### Frontend — Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Point to `frontend/app.py`
3. Set `BACKEND_URL` to your Render backend URL

---

## Performance Metrics

| Metric | Typical Value |
|---|---|
| Retrieval (10 chunks) | ~200ms |
| CrossEncoder rerank | ~150ms |
| Gemini generation | ~1-3s |
| End-to-end latency | ~2-4s |
| RAGAS faithfulness | 0.75-0.90 (domain-dependent) |

---

## Testing

```bash
pytest tests/ -v
# 39 tests covering auth, API, RAG, feedback, evaluation, memory
```

---

## Screenshots

> Add screenshots of Chat UI, Analytics Dashboard, RAG Performance page, and Admin panel here after deployment.

---

## License

MIT — Built for educational and portfolio purposes.
