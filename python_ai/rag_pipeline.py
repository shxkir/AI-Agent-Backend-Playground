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
from typing import Dict, List, Optional, Tuple

import anthropic
import chromadb
import huggingface_hub

# Compatibility shim for deprecated huggingface_hub.cached_download.
if not hasattr(huggingface_hub, "cached_download"):
    from huggingface_hub import hf_hub_download

    def cached_download(*args, **kwargs):
        return hf_hub_download(*args, **kwargs)

    huggingface_hub.cached_download = cached_download

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

    def generate_answer(
        self, question: str, top_k: int = 4
    ) -> Tuple[str, List[Dict[str, str]]]:
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
            citations: List[Dict[str, str]] = []
        else:
            citations = []
            for metadata, doc_id, doc_text in zip(metadatas, ids, documents):
                metadata = metadata or {}
                source = metadata.get("source") or doc_id
                citations.append({"source": source, "text": doc_text})
            context_blocks = [
                f"[{index + 1}] Source: {citation['source']}\n{citation['text']}"
                for index, citation in enumerate(citations)
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

    def delete_document(self, doc_id: str) -> None:
        """Remove a document from the vector store."""
        existing = self._collection.get(ids=[doc_id])
        if not existing.get("ids"):
            raise ValueError(f"Document with id '{doc_id}' does not exist.")
        self._collection.delete(ids=[doc_id])

    def update_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        """Update an existing document's contents and metadata."""
        if not text.strip():
            raise ValueError("Updated document text cannot be empty.")

        result = self._collection.get(ids=[doc_id], include=["metadatas"])
        if not result.get("ids"):
            raise ValueError(f"Document with id '{doc_id}' does not exist.")

        current_metadata = (result.get("metadatas") or [{}])[0] or {}
        # Ensure the doc_id remains stored with the record.
        merged_metadata = {**current_metadata, "doc_id": doc_id}
        if metadata:
            merged_metadata.update(metadata)

        embedding = self._embedder.encode([text], convert_to_numpy=True)[0].tolist()

        self._collection.update(
            ids=[doc_id],
            documents=[text],
            metadatas=[merged_metadata],
            embeddings=[embedding],
        )


_PIPELINE = RagPipeline()


def add_document(text: str, metadata: Dict[str, str] | None = None) -> str:
    """Module-level helper to add a document to the vector store."""
    return _PIPELINE.add_document(text=text, metadata=metadata)


def generate_answer(question: str, top_k: int = 4) -> Tuple[str, List[Dict[str, str]]]:
    """Module-level helper to generate an answer and citations for a query."""
    return _PIPELINE.generate_answer(question=question, top_k=top_k)


def delete_document(doc_id: str) -> None:
    """Delete a document from the store."""
    _PIPELINE.delete_document(doc_id=doc_id)


def update_document(
    doc_id: str,
    text: str,
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    """Update a document in the store."""
    _PIPELINE.update_document(doc_id=doc_id, text=text, metadata=metadata)
