"""
Retrieval-augmented generation utilities used by the FastAPI microservice.

The pipeline persists documents in a local ChromaDB vector store using
`sentence-transformers/all-MiniLM-L6-v2` embeddings.  When a question is asked,
the most relevant documents are retrieved and passed to Anthropic's Claude API
to craft an answer grounded in the stored context.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

import anthropic
import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


load_dotenv()

_DB_DIR = Path(__file__).resolve().parent / "chroma_db"
_DB_DIR.mkdir(exist_ok=True)


class RagPipeline:
    """Lightweight RAG orchestrator backed by ChromaDB and Claude."""

    def __init__(self) -> None:
        self._embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self._anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self._anthropic_client: anthropic.Anthropic | None = None
        self._client = chromadb.PersistentClient(
            path=str(_DB_DIR),
            settings=Settings(
                anonymized_telemetry=False,
                is_persistent=True,
                allow_reset=True,
            ),
        )
        self._collection: Collection = self._client.get_or_create_collection(
            name="documents"
        )

    def _ensure_llm(self) -> anthropic.Anthropic:
        if not self._anthropic_key:
            raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set.")
        if self._anthropic_client is None:
            self._anthropic_client = anthropic.Anthropic(api_key=self._anthropic_key)
        return self._anthropic_client

    def add_document(self, text: str, metadata: Dict[str, str] | None = None) -> str:
        """Store a document in the vector store and return its generated ID."""
        if not text.strip():
            raise ValueError("Document text cannot be empty.")

        doc_id = str(uuid.uuid4())
        metadata = metadata or {}
        embedding = self._embedder.encode([text], convert_to_numpy=True)[0].tolist()

        self._collection.add(
            documents=[text],
            metadatas=[{**metadata, "doc_id": doc_id}],
            ids=[doc_id],
            embeddings=[embedding],
        )
        return doc_id

    def generate_answer(self, question: str, top_k: int = 4) -> Tuple[str, List[str]]:
        """Retrieve supporting documents and ask Claude for a grounded answer."""
        if not question.strip():
            raise ValueError("Query cannot be empty.")

        query_embedding = self._embedder.encode([question], convert_to_numpy=True)[
            0
        ].tolist()

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=max(1, top_k),
            include=["documents", "metadatas", "ids", "distances"],
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        ids = results.get("ids", [[]])[0]

        if not documents:
            context_blocks = ["No documents matched the query."]
            citations: List[str] = []
        else:
            citations = [
                (metadata or {}).get("source") or doc_id
                for metadata, doc_id in zip(metadatas, ids)
            ]
            context_blocks = [
                f"[{index + 1}] Source: {citation}\n{doc}"
                for index, (citation, doc) in enumerate(zip(citations, documents))
            ]

        context = "\n\n".join(context_blocks)
        prompt = (
            "You are an AI assistant that answers questions using the provided "
            "context. Only rely on the context when possible and include short "
            "citations in the format [source]. If the context is insufficient, "
            "state that you do not know.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )

        llm = self._ensure_llm()
        response = llm.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=400,
            temperature=0.2,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        )

        answer = "".join(
            block.text for block in response.content if hasattr(block, "text")
        ).strip()

        return answer, citations


_PIPELINE = RagPipeline()


def add_document(text: str, metadata: Dict[str, str] | None = None) -> str:
    """Module-level helper to add a document to the vector store."""
    return _PIPELINE.add_document(text=text, metadata=metadata)


def generate_answer(question: str, top_k: int = 4) -> Tuple[str, List[str]]:
    """Module-level helper to generate an answer and citations for a query."""
    return _PIPELINE.generate_answer(question=question, top_k=top_k)
