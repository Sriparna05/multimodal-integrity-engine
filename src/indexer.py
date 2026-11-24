"""Vector indexing engine using sentence-transformers and FAISS IndexFlatIP."""

import logging
from typing import Optional, Union

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

from src.models import TextChunk

logger = logging.getLogger(__name__)

# Cached SentenceTransformer instance
_EMBEDDING_MODEL: Optional[SentenceTransformer] = None


def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    """Lazily loads and returns the SentenceTransformer model on CUDA or CPU."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading embedding model '{model_name}' on {device}...")
        _EMBEDDING_MODEL = SentenceTransformer(model_name, device=device)
    return _EMBEDDING_MODEL


class VectorIndex:
    """Manages an in-memory FAISS IndexFlatIP for exact cosine similarity searches."""

    def __init__(self, name: str = "default_index", model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.name = name
        self.model = get_embedding_model(model_name)
        self.chunks: list[TextChunk] = []
        self.index: Optional[faiss.IndexFlatIP] = None
        self.dimension: Optional[int] = None

    def add_chunks(self, chunks: list[TextChunk]) -> None:
        """Encodes TextChunks into normalized float32 embeddings and adds them to FAISS."""
        if not chunks:
            return

        texts = [c.text for c in chunks]
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

        if self.index is None:
            self.dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(self.dimension)

        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(
        self,
        query: Union[str, TextChunk, np.ndarray],
        top_k: int = 5,
    ) -> list[tuple[TextChunk, float]]:
        """Searches the index for chunks closest to the query by cosine similarity.

        Returns:
            List of (TextChunk, cosine_similarity_score) tuples sorted in descending order.
        """
        if self.index is None or len(self.chunks) == 0:
            return []

        if isinstance(query, np.ndarray):
            query_vec = query.astype(np.float32)
            if query_vec.ndim == 1:
                query_vec = np.expand_dims(query_vec, axis=0)
        else:
            query_text = query.text if isinstance(query, TextChunk) else query
            query_vec = self.model.encode(
                [query_text],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype(np.float32)

        actual_k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(query_vec, actual_k)

        matches: list[tuple[TextChunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.chunks):
                matches.append((self.chunks[idx], float(score)))

        return matches

    def __len__(self) -> int:
        return len(self.chunks)
