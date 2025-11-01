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

- `rust_api/` – Actix-Web gateway (`src/main.rs`) that enforces the `X-API-KEY` header, proxies `/api/ask` and `/api/add_doc` to Python, retries failures with exponential backoff, and exposes `/api/health`.
- `python_ai/` – FastAPI service (`app.py`) with a styled landing page, the public REST endpoints, and RAG helpers in `rag_pipeline.py`. Documents are stored in a persistent ChromaDB directory (`python_ai/chroma_db/`) and embeddings come from `sentence-transformers/all-MiniLM-L6-v2`.
- `scripts/` – `run_python.sh` bootstraps a virtual environment, installs FastAPI + LangChain/LangGraph dependencies, and launches Uvicorn; `run_rust.sh` starts the Actix gateway.
- `docs/` – Reference material (`architecture.md`) describing the Rust↔Python message flow, reliability patterns, and roadmap items.
- `langgraph/` – Example LangGraph state machine (`graph.py`) that classifies intent, retrieves context, and generates answers. Use it as a scaffold for future agent orchestration work.

---
## 🛠️ Prerequisites

- **Rust toolchain** with `cargo` (install via [rustup](https://rustup.rs/)).
- **Python 3.10+** plus `pip`. The project installs dependencies from `python_ai/requirements.txt`.
- **Anthropic access** (or a compatible Claude API key) for live answer generation. Without a key the service will raise `ANTHROPIC_API_KEY` errors when the pipeline is invoked.
- Optional: a [Hugging Face](https://huggingface.co/) account if you want to cache the embedding model ahead of time; otherwise it downloads on first run.

---
## 🔑 Configuration

Set these environment variables before launching services:

- `ANTHROPIC_API_KEY` – Required by the Python RAG pipeline to call Claude.
- `PYTHON_AI_URL` – Optional override used by the Rust gateway to locate the FastAPI service (`http://127.0.0.1:8001` by default).
- `RUST_API_PORT` – Optional port for the Actix server (defaults to `8000`).

The project intentionally omits `.env` files from version control—export secrets in your shell or use a local `.env` that stays untracked.

---
## 🗃️ Data & Caches

- ChromaDB state persists under `python_ai/chroma_db/`. Remove this directory to clear stored documents and embeddings.
- Hugging Face and sentence-transformer downloads are cached under your user cache directory (typically `~/.cache`). Clear them if you need a fresh model pull.
- No virtual environments or `.env` files are committed; run `rm -rf python_ai/.venv rust_api/target` to reset local builds before publishing.

---
## ⚙️ Developer Utilities

- Format Rust code with `cargo fmt` and Python code with `python3 -m black python_ai langgraph`.
- `scripts/run_python.sh` and `scripts/run_rust.sh` wrap the standard startup flow so you can launch both services from the repo root.
- `docs/architecture.md` captures the higher-level system design, security considerations, and future roadmap.
- `langgraph/graph.py` demonstrates how to wire intent classification, retrieval, and answer generation into a LangGraph state machine—use `python langgraph/graph.py` to inspect the stubbed flow.
- Automated tests are not yet included; validate changes by exercising the API via `curl` or the FastAPI `/docs` explorer.

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
