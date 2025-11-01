# Architecture Overview

EdgeLink is designed to separate concerns between a high‑performance HTTP
gateway and a flexible AI microservice.  The gateway, implemented in Rust using
Actix, handles request parsing, authentication, rate limiting and metrics
collection.  It delegates intelligent behaviour to a Python service built on
FastAPI, LangChain【879956037759048†L120-L175】 and LangGraph【389737802864832†L76-L124】.

## Components

### Rust Gateway

The Rust component exposes endpoints under `/api/*`.  Each handler receives a
JSON payload, deserializes it and forwards it to the Python AI layer.  Responses
are serialized back to JSON and include timing information.  The gateway is
responsible for enforcing JSON Web Token (JWT) authentication and can be
extended with rate limiting and logging middleware.  Because it is built in
Rust, it offers strong memory safety and predictable performance.

### Python AI Service

The AI layer is a FastAPI application that implements retrieval‑augmented
generation (RAG).  Incoming questions are processed by `rag_pipeline.py` which
will, once implemented, perform the following steps:

1. **Document ingestion** – Convert and chunk user provided documents and load
   them into a vector store such as ChromaDB.
2. **Embedding and retrieval** – Use an embedding model to embed both
   questions and document chunks.  Retrieve the top_k most similar chunks for
   a given query.
3. **LLM invocation** – Pass the retrieved context along with the original
   question to a large language model (Anthropic Claude or OpenAI GPT) to
   produce an answer.  Extract citations referring back to the retrieved
   chunks.

The service also provides a health check endpoint for liveness probes.

### LangGraph Workflow

LangGraph is used to define agentic workflows as state machines.  A simple
example graph is included in `langgraph/graph.py`.  It illustrates how you
might classify the user’s intent (question answering vs. task execution),
retrieve context and generate an answer.  In a full implementation you would
replace these nodes with calls to the RAG pipeline, code generation agents,
testing harnesses and security filters.

## Deployment

To deploy EdgeLink, run the Rust gateway and Python AI service in separate
processes or containers.  Both services can be containerized using Docker and
orchestrated with docker‑compose or Kubernetes.  The Rust service listens on
port 8000 by default, while the Python service listens on port 8001.  The
gateway can be configured via environment variables to point to the correct
Python service address.

## Security Considerations

- **Authentication** – Use JWTs or another mechanism to ensure only
  authorized clients can call the API.
- **Rate limiting** – Protect the service from abuse by throttling requests.
- **Post‑quantum cryptography** – Integrate PQC algorithms such as Kyber512 to
  secure key exchange in anticipation of quantum attacks.

## Roadmap

Future iterations of EdgeLink could include:

- Automatic code generation and testing via agentic workflows.
- Integration with additional vector stores and embedding models.
- Support for streaming responses and server‑sent events.
- Deployment templates for cloud environments.
