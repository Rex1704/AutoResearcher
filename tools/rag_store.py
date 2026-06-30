"""
tools/rag_store.py
Lightweight FAISS vector store used to ground the final report in retrieved
evidence (prevents hallucination, enables citation lookup).
"""
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


def _chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + chunk_size]))
        i += chunk_size - overlap
    return [c for c in chunks if c.strip()]


class EvidenceStore:
    """In-memory FAISS index over scraped evidence chunks for one research run."""

    def __init__(self):
        self.model = _get_model()
        self.index = None
        self.chunks: list[str] = []
        self.meta: list[dict] = []  # {url, title} per chunk

    def add_evidence(self, evidence: list[dict]):
        for item in evidence:
            for chunk in _chunk_text(item["content"]):
                self.chunks.append(chunk)
                self.meta.append({"url": item["url"], "title": item["title"]})

        if not self.chunks:
            return

        embeddings = self.model.encode(self.chunks, convert_to_numpy=True, normalize_embeddings=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))

    def query(self, question: str, k: int = 5) -> list[dict]:
        if self.index is None or not self.chunks:
            return []
        q_emb = self.model.encode([question], convert_to_numpy=True, normalize_embeddings=True)
        scores, idxs = self.index.search(q_emb.astype(np.float32), min(k, len(self.chunks)))
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append({
                "text": self.chunks[idx],
                "url": self.meta[idx]["url"],
                "title": self.meta[idx]["title"],
                "score": float(score),
            })
        return results
