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
