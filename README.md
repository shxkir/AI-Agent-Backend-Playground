# AI Agent Backend Playground

Created by Ismaiel Shakir.

This project is my learning and experimentation environment for building modern AI backend systems.

It includes a microservice architecture using:
- **Rust (Actix-Web)** as a high-performance gateway API
- **Python (FastAPI)** as the AI service layer
- Foundations for **RAG (Retrieval-Augmented Generation)**
- **LangChain & LangGraph** concepts for agent workflows

## 🧠 Objective

Learn the fundamentals behind production-grade AI systems and build a working base that can be expanded into a full agentic system.

This playground helped me understand:
###  Rust
- API routing
- Ownership & borrowing basics
- Async backend fundamentals

### FastAPI
- Async endpoint handling
- Swagger UI testing
- Microservice communication

### AI Concepts
- RAG pipeline (retrieve → inject context → generate)
- Vector embeddings and semantic search
- Agent workflows & tool use
- Claude for coding assistance & reasoning

---
## 📂 Project Structure

- `rust_api/` – Actix-Web gateway
- `python_ai/` – FastAPI AI microservice with RAG helpers
- `scripts/` – Convenience launch scripts for local development
- `docs/` – Architecture notes and design references that document the gateway↔AI flow
- `langgraph/` – Workspace for future LangGraph-based orchestration experiments

---
## 🚀 Run the Services Locally

### 1. Start the FastAPI AI microservice

```bash
cd python_ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key-here"
uvicorn app:app --port 8001 --reload
```

> The helper script `scripts/run_python.sh` wraps these steps and ensures dependencies are installed.

### 2. Start the Rust gateway

```bash
cd rust_api
cargo run
```

The gateway listens on port `8000` by default and forwards `/api/ask` requests to the FastAPI service (configured via the `PYTHON_AI_URL` env var, defaulting to `http://127.0.0.1:8001`).

> You can also call `scripts/run_rust.sh` from the repository root to launch the gateway.

---
## 🔌 Gateway Capabilities

- `/api/ask` and `/api/add_doc` are proxied to FastAPI with JSON logging, latency tracking, and three retry attempts (exponential backoff) for resiliency.
- Every gateway call must include a non-empty `X-API-KEY` header; supply any value during development (e.g. `-H "X-API-KEY: dev-key"`).
- Health checks are exposed on `/api/health`, which also powers the landing page indicator.

### Sample gateway requests

```bash
# Ingest a document
curl -X POST http://127.0.0.1:8000/api/add_doc \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: dev-key" \
  -d '{ "text": "EdgeLink secures the gateway with Actix.", "metadata": { "source": "playbook" } }'

# Ask a question
curl -X POST http://127.0.0.1:8000/api/ask \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: dev-key" \
  -d '{ "query": "How does EdgeLink secure the gateway?", "top_k": 4 }'
```

---
## 🧾 FastAPI Endpoints

The AI microservice now exposes:

- `POST /ask` – Returns grounded answers with citations that include the retrieved text chunks. Failures return `{ "answer": "Service busy — try again soon!", "citations": [] }`.
- `POST /add_doc` – Adds documents into Chroma with optional metadata.
- `DELETE /delete_doc` – Removes a document by id (returns 404 if it does not exist).
- `PUT /update_doc` – Rewrites document contents and metadata while keeping embeddings fresh.
- `GET /health` – Used by the gateway and landing page status indicator.

The landing page (`/`) now includes a simple chat box wired to `/ask`, live health status derived from `/health`, and updated quickstart curl snippets.
