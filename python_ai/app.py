"""
FastAPI application that exposes the AI endpoints for EdgeLink.

This service receives requests from the Rust gateway and performs
retrieval‑augmented generation (RAG) to answer questions over indexed data.
"""

import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict, List, Optional

from rag_pipeline import add_document, delete_document, generate_answer, update_document

logger = logging.getLogger(__name__)

app = FastAPI(
    title="EdgeLink AI Service",
    description="Provides AI-powered question answering and automation capabilities.",
    version="0.1.0",
)


@app.get("/", response_class=HTMLResponse)
async def home() -> HTMLResponse:
    """Serve a lightweight landing page so stakeholders can explore the API."""
    return HTMLResponse(
        content="""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1.0" />
            <title>EdgeLink AI Backend</title>
            <link rel="preconnect" href="https://fonts.googleapis.com">
            <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
            <style>
                :root {
                    color-scheme: light dark;
                    --bg: #0f172a;
                    --surface: rgba(15, 23, 42, 0.8);
                    --text: #e2e8f0;
                    --accent: #38bdf8;
                    --accent-dark: #0284c7;
                    --card: rgba(30, 41, 59, 0.8);
                }
                body {
                    margin: 0;
                    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                    background: radial-gradient(circle at top, #0ea5e9, #020617 65%);
                    color: var(--text);
                    min-height: 100vh;
                    display: flex;
                    flex-direction: column;
                }
                header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 1.5rem clamp(1.5rem, 4vw, 3.5rem);
                }
                .brand {
                    display: flex;
                    align-items: center;
                    gap: 0.75rem;
                    font-weight: 600;
                    letter-spacing: 0.05em;
                }
                .brand span {
                    font-size: 1.05rem;
                }
                nav a {
                    margin-left: 1.5rem;
                    text-decoration: none;
                    color: var(--text);
                    font-weight: 500;
                    position: relative;
                }
                nav a::after {
                    content: "";
                    position: absolute;
                    left: 0;
                    bottom: -0.35rem;
                    width: 100%;
                    height: 2px;
                    background: linear-gradient(90deg, var(--accent), var(--accent-dark));
                    transform: scaleX(0);
                    transform-origin: left;
                    transition: transform 180ms ease;
                }
                nav a:hover::after {
                    transform: scaleX(1);
                }
                main {
                    flex: 1;
                    display: grid;
                    gap: clamp(1.5rem, 5vw, 3rem);
                    padding: 0 clamp(1.5rem, 4vw, 4rem) 4rem;
                }
                .hero {
                    display: grid;
                    gap: 2rem;
                    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                    align-items: center;
                    background: var(--surface);
                    border-radius: 22px;
                    padding: clamp(2rem, 4vw, 3.5rem);
                    border: 1px solid rgba(148, 163, 184, 0.2);
                    box-shadow: 0 30px 60px rgba(15, 23, 42, 0.35);
                }
                .hero h1 {
                    font-size: clamp(2rem, 4vw, 3rem);
                    margin: 0;
                    line-height: 1.1;
                }
                .hero p {
                    margin: 1rem 0 2rem;
                    color: rgba(226, 232, 240, 0.85);
                    max-width: 32rem;
                }
                .cta-group {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 0.75rem;
                }
                .cta {
                    border: none;
                    border-radius: 999px;
                    padding: 0.85rem 1.4rem;
                    font-weight: 600;
                    cursor: pointer;
                    transition: transform 150ms ease, box-shadow 150ms ease;
                }
                .cta.primary {
                    background: linear-gradient(135deg, var(--accent), var(--accent-dark));
                    color: #0b1120;
                }
                .cta.secondary {
                    background: rgba(148, 163, 184, 0.15);
                    color: var(--text);
                    border: 1px solid rgba(148, 163, 184, 0.25);
                }
                .cta:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 12px 24px rgba(56, 189, 248, 0.25);
                }
                .metrics {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
                    gap: 1rem;
                }
                .metric-card {
                    background: rgba(15, 23, 42, 0.55);
                    border-radius: 18px;
                    padding: 1.25rem;
                    border: 1px solid rgba(56, 189, 248, 0.15);
                    transition: transform 200ms ease, border-color 200ms ease;
                }
                .metric-card:hover {
                    transform: translateY(-4px);
                    border-color: rgba(56, 189, 248, 0.45);
                }
                .metric-card span {
                    display: block;
                    font-size: 2rem;
                    font-weight: 600;
                    color: var(--accent);
                }
                .metric-card small {
                    text-transform: uppercase;
                    letter-spacing: 0.08em;
                    color: rgba(148, 163, 184, 0.85);
                }
                .panel {
                    background: var(--card);
                    border-radius: 20px;
                    padding: clamp(1.5rem, 3vw, 2.25rem);
                    border: 1px solid rgba(148, 163, 184, 0.15);
                    backdrop-filter: blur(18px);
                }
                .panel h2 {
                    margin-top: 0;
                    font-size: 1.4rem;
                }
                .features {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
                    gap: 1.5rem;
                    margin-top: 1.5rem;
                }
                .feature-card {
                    padding: 1.5rem;
                    border-radius: 16px;
                    border: 1px solid rgba(148, 163, 184, 0.18);
                    background: rgba(15, 23, 42, 0.45);
                    transition: transform 200ms ease, border-color 200ms ease;
                }
                .feature-card:hover {
                    transform: translateY(-6px);
                    border-color: rgba(56, 189, 248, 0.4);
                }
                .feature-card h3 {
                    margin-top: 0;
                }
                .console {
                    margin-top: 1.5rem;
                    background: rgba(2, 6, 23, 0.8);
                    border-radius: 14px;
                    padding: 1.25rem;
                    font-family: 'JetBrains Mono', 'SFMono-Regular', Menlo, monospace;
                    font-size: 0.95rem;
                    color: rgba(203, 213, 225, 0.95);
                    border: 1px solid rgba(15, 118, 110, 0.35);
                    position: relative;
                    overflow-x: auto;
                }
                .console::before {
                    content: "EdgeLink CLI";
                    position: absolute;
                    top: -12px;
                    left: 16px;
                    font-size: 0.75rem;
                    letter-spacing: 0.08em;
                    background: rgba(15, 23, 42, 0.85);
                    padding: 0.15rem 0.5rem;
                    border-radius: 999px;
                    border: 1px solid rgba(148, 163, 184, 0.45);
                }
                .status-row {
                    display: flex;
                    align-items: center;
                    gap: 0.75rem;
                    margin-top: 1.5rem;
                }
                .status-label {
                    font-size: 0.85rem;
                    letter-spacing: 0.08em;
                    text-transform: uppercase;
                    color: rgba(148, 163, 184, 0.75);
                }
                .status-indicator {
                    display: inline-flex;
                    align-items: center;
                    gap: 0.55rem;
                    background: rgba(15, 23, 42, 0.6);
                    border-radius: 999px;
                    padding: 0.35rem 0.85rem;
                    border: 1px solid rgba(148, 163, 184, 0.25);
                    font-weight: 600;
                }
                .status-dot {
                    width: 10px;
                    height: 10px;
                    border-radius: 999px;
                    background: #f87171;
                    box-shadow: 0 0 10px rgba(248, 113, 113, 0.45);
                    transition: background 180ms ease, box-shadow 180ms ease;
                }
                .status-indicator.up .status-dot {
                    background: #34d399;
                    box-shadow: 0 0 12px rgba(52, 211, 153, 0.5);
                }
                .chat-panel {
                    margin-top: 2rem;
                    display: grid;
                    gap: 1rem;
                }
                .chat-form {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 0.85rem;
                }
                .chat-form input[type="text"] {
                    flex: 1;
                    min-width: 260px;
                    border-radius: 14px;
                    border: 1px solid rgba(148, 163, 184, 0.3);
                    padding: 0.85rem 1.15rem;
                    background: rgba(15, 23, 42, 0.6);
                    color: var(--text);
                    font-size: 1rem;
                }
                .chat-form input[type="text"]:focus {
                    outline: none;
                    border-color: rgba(56, 189, 248, 0.6);
                    box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.15);
                }
                .chat-form .ask-button {
                    flex: 0 0 auto;
                    padding: 0.85rem 1.6rem;
                }
                .chat-response {
                    background: rgba(2, 6, 23, 0.75);
                    border: 1px solid rgba(56, 189, 248, 0.25);
                    border-radius: 16px;
                    padding: 1.25rem;
                    display: grid;
                    gap: 0.75rem;
                }
                .chat-response h3 {
                    margin: 0;
                }
                .chat-response h4 {
                    margin: 0.25rem 0 0.5rem;
                    color: rgba(148, 163, 184, 0.85);
                    font-size: 0.95rem;
                    letter-spacing: 0.04em;
                }
                .citations-list {
                    list-style: none;
                    margin: 0;
                    padding: 0;
                    display: grid;
                    gap: 0.65rem;
                }
                .citation-item {
                    border-radius: 12px;
                    border: 1px solid rgba(148, 163, 184, 0.25);
                    background: rgba(15, 23, 42, 0.6);
                    padding: 0.75rem 1rem;
                }
                .citation-item strong {
                    display: block;
                    font-size: 0.8rem;
                    letter-spacing: 0.08em;
                    text-transform: uppercase;
                    color: rgba(56, 189, 248, 0.9);
                    margin-bottom: 0.35rem;
                }
                footer {
                    text-align: center;
                    padding: 2rem 1rem 3rem;
                    color: rgba(148, 163, 184, 0.75);
                    font-size: 0.9rem;
                }
                @media (max-width: 640px) {
                    header {
                        flex-direction: column;
                        gap: 1rem;
                    }
                    nav a {
                        margin: 0 0.75rem;
                    }
                    .cta-group {
                        width: 100%;
                    }
                    .cta {
                        flex: 1;
                    }
                    .chat-form {
                        flex-direction: column;
                    }
                    .chat-form .ask-button {
                        width: 100%;
                        justify-content: center;
                    }
                }
            </style>
        </head>
        <body>
            <header>
                <div class="brand">
                    <img src="https://em-content.zobj.net/source/apple/354/rocket_1f680.png" width="32" height="32" alt="Rocket icon" />
                    <span>EdgeLink AI Backend</span>
                </div>
                <nav>
                    <a href="/docs">API Explorer</a>
                    <a href="https://swagger.io/" target="_blank" rel="noreferrer">Swagger</a>
                    <a href="https://langchain.com/" target="_blank" rel="noreferrer">LangChain</a>
                </nav>
            </header>
            <main>
                <section class="hero">
                    <div>
                        <h1>Bring EdgeUp&rsquo;s Knowledge Graph to Life</h1>
                        <p>EdgeLink orchestrates retrieval-augmented generation, task automation, and secure workflows. Connect your data, plug in your favorite LLM, and deliver answers with real citations.</p>
                        <div class="cta-group">
                            <button class="cta primary" onclick="window.location.href='/docs'">Launch API Console</button>
                            <button class="cta secondary" onclick="toggleConsole()">Show Quickstart</button>
                        </div>
                    </div>
                    <div class="metrics">
                        <article class="metric-card">
                            <span id="latency">42&nbsp;ms</span>
                            <small>Median Latency (Stub)</small>
                        </article>
                        <article class="metric-card">
                            <span>99.9%</span>
                            <small>Gateway Uptime</small>
                        </article>
                        <article class="metric-card">
                            <span>4</span>
                            <small>Active Pipelines</small>
                        </article>
                    </div>
                </section>
                <section class="panel">
                    <h2>What can the EdgeLink API do?</h2>
                    <div class="features">
                        <article class="feature-card">
                            <h3>RAG Answers</h3>
                            <p>Query your knowledge base with grounded explanations and citation trails.</p>
                        </article>
                        <article class="feature-card">
                            <h3>Ingestion</h3>
                            <p>Stream documents into the vector store with schema-aware chunking.</p>
                        </article>
                        <article class="feature-card">
                            <h3>Automation</h3>
                            <p>Trigger LangGraph automations for QA checks, code generation, and more.</p>
                        </article>
                        <article class="feature-card">
                            <h3>Secure by Default</h3>
                            <p>JWT-ready gateway built on Actix with room for PQ crypto hardening.</p>
                        </article>
                    </div>
                    <div id="console" class="console" style="display: none;">
<pre><code>$ curl -X POST http://127.0.0.1:8000/api/add_doc \
    -H "Content-Type: application/json" \
    -H "X-API-KEY: dev-key" \
    -d '{ "text": "EdgeLink secures the gateway with Actix.", "metadata": { "source": "playbook" } }'

$ curl -X POST http://127.0.0.1:8000/api/ask \
    -H "Content-Type: application/json" \
    -H "X-API-KEY: dev-key" \
    -d '{ "query": "How does EdgeLink secure the gateway?", "top_k": 4 }'

$ curl -X DELETE http://127.0.0.1:8001/delete_doc \
    -H "Content-Type: application/json" \
    -d '{ "document_id": "abc-123" }'</code></pre>
                    </div>
                    <div class="chat-panel">
                        <div class="status-row">
                            <span class="status-label">Service Status</span>
                            <span id="status-indicator" class="status-indicator">
                                <span class="status-dot"></span>
                                <span id="status-text">Checking...</span>
                            </span>
                        </div>
                        <form id="chat-form" class="chat-form">
                            <input id="chat-input" type="text" placeholder="Ask a question" aria-label="Ask a question" required />
                            <button type="submit" class="cta primary ask-button">Ask EdgeLink</button>
                        </form>
                        <div id="chat-response" class="chat-response" hidden>
                            <h3>Answer</h3>
                            <p id="answer-text">Ready to help—ask away!</p>
                            <div id="citations-block" hidden>
                                <h4>Citations</h4>
                                <ul id="citations-list" class="citations-list"></ul>
                            </div>
                        </div>
                    </div>
                </section>
            </main>
            <footer>
                &copy; {year} EdgeUp • Built with FastAPI, LangChain, and Actix.
            </footer>
            <script>
                const metrics = [32, 48, 56, 41];
                let pointer = 0;
                setInterval(() => {
                    pointer = (pointer + 1) % metrics.length;
                    document.getElementById('latency').innerHTML = metrics[pointer] + '&nbsp;ms';
                }, 2400);

                function toggleConsole() {
                    const el = document.getElementById('console');
                    el.style.display = el.style.display === 'none' ? 'block' : 'none';
                }

                document.querySelector('footer').innerHTML =
                    document.querySelector('footer').innerHTML.replace('{year}', new Date().getFullYear());

                const statusIndicator = document.getElementById('status-indicator');
                const statusText = document.getElementById('status-text');
                const statusDot = statusIndicator ? statusIndicator.querySelector('.status-dot') : null;

                function setStatus(up) {
                    if (!statusIndicator || !statusDot || !statusText) return;
                    statusIndicator.classList.toggle('up', up);
                    statusText.textContent = up ? 'Online' : 'Offline';
                }

                async function refreshHealth() {
                    if (!statusIndicator) return;
                    try {
                        const resp = await fetch('/health');
                        setStatus(resp.ok);
                    } catch (error) {
                        setStatus(false);
                    }
                }

                refreshHealth();
                setInterval(refreshHealth, 8000);

                const chatForm = document.getElementById('chat-form');
                const chatInput = document.getElementById('chat-input');
                const chatResponse = document.getElementById('chat-response');
                const answerText = document.getElementById('answer-text');
                const citationsBlock = document.getElementById('citations-block');
                const citationsList = document.getElementById('citations-list');
                const askButton = chatForm ? chatForm.querySelector('button[type="submit"]') : null;

                if (chatForm && chatInput && chatResponse && answerText && citationsBlock && citationsList && askButton) {
                    const defaultButtonLabel = askButton.textContent;
                    chatForm.addEventListener('submit', async (event) => {
                        event.preventDefault();
                        const query = chatInput.value.trim();
                        if (!query) {
                            chatInput.focus();
                            return;
                        }

                        askButton.disabled = true;
                        askButton.textContent = 'Thinking...';
                        answerText.textContent = 'Thinking...';
                        chatResponse.hidden = false;
                        citationsBlock.hidden = true;

                        try {
                            const response = await fetch('/ask', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ query, top_k: 4 })
                            });
                            const data = await response.json().catch(() => ({}));
                            if (!response.ok) {
                                throw new Error(data.detail || 'Request failed');
                            }
                            answerText.textContent = data.answer || 'No answer returned.';

                            const citations = Array.isArray(data.citations) ? data.citations : [];
                            citationsList.innerHTML = '';
                            if (citations.length > 0) {
                                citations.forEach((citation, index) => {
                                    const li = document.createElement('li');
                                    li.className = 'citation-item';
                                    const source = citation.source || `Source ${index + 1}`;
                                    const text = citation.text || '';
                                    li.innerHTML = `<strong>${source}</strong><span>${text}</span>`;
                                    citationsList.appendChild(li);
                                });
                                citationsBlock.hidden = false;
                            } else {
                                citationsBlock.hidden = true;
                            }
                        } catch (error) {
                            answerText.textContent = 'Service busy — try again soon!';
                            citationsBlock.hidden = true;
                        } finally {
                            askButton.disabled = false;
                            askButton.textContent = defaultButtonLabel;
                        }
                    });
                }
            </script>
        </body>
        </html>
        """,
        status_code=200,
    )


class Citation(BaseModel):
    source: str
    text: str


class AskRequest(BaseModel):
    query: str
    top_k: Optional[int] = 4


class AskResponse(BaseModel):
    answer: str
    citations: List[Citation]


class AddDocRequest(BaseModel):
    text: str
    metadata: Optional[Dict[str, str]] = None


class AddDocResponse(BaseModel):
    document_id: str


class DeleteDocRequest(BaseModel):
    document_id: str


class DeleteDocResponse(BaseModel):
    document_id: str
    status: str = "deleted"


class UpdateDocRequest(BaseModel):
    document_id: str
    text: str
    metadata: Optional[Dict[str, str]] = None


class UpdateDocResponse(BaseModel):
    document_id: str
    status: str = "updated"


@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(req: AskRequest) -> AskResponse:
    """
    Accepts a question and returns an answer grounded in the indexed knowledge.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        answer, citation_dicts = generate_answer(req.query, req.top_k or 4)
        citations = [Citation(**citation) for citation in citation_dicts]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # broad catch to provide a graceful fallback
        logger.exception("RAG pipeline failed", exc_info=exc)
        return AskResponse(answer="Service busy — try again soon!", citations=[])

    return AskResponse(answer=answer, citations=citations)


@app.get("/health")
async def healthcheck() -> str:
    """
    Simple healthcheck endpoint used by the Rust gateway to ensure the
    microservice is responsive.
    """
    return "OK"


@app.post("/add_doc", response_model=AddDocResponse)
async def add_document_endpoint(req: AddDocRequest) -> AddDocResponse:
    """Ingest a document into the local vector database."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Document text cannot be empty")
    try:
        document_id = add_document(req.text, req.metadata)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return AddDocResponse(document_id=document_id)


@app.delete("/delete_doc", response_model=DeleteDocResponse)
async def delete_document_endpoint(req: DeleteDocRequest) -> DeleteDocResponse:
    """Remove a document from the vector store."""
    if not req.document_id.strip():
        raise HTTPException(status_code=400, detail="Document ID cannot be empty")
    try:
        delete_document(req.document_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return DeleteDocResponse(document_id=req.document_id)


@app.put("/update_doc", response_model=UpdateDocResponse)
async def update_document_endpoint(req: UpdateDocRequest) -> UpdateDocResponse:
    """Update the stored contents of a document."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Document text cannot be empty")
    try:
        update_document(req.document_id, req.text, req.metadata)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return UpdateDocResponse(document_id=req.document_id)
