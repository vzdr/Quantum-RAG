"""
Embedding generation using sentence-transformers.
"""
from typing import List
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from .data_models import Chunk, EmbeddedChunk

class EmbeddingGenerator:
    """A wrapper for sentence-transformers models."""
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)

    def embed(self, texts: List[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        """Embeds a list of texts."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )

    def embed_chunks(self, chunks: List[Chunk], **kwargs) -> List[EmbeddedChunk]:
        """Embeds a list of Chunks."""
        embeddings = self.embed([chunk.text for chunk in chunks], **kwargs)
        return [EmbeddedChunk(chunk=chunk, embedding=emb) for chunk, emb in zip(chunks, embeddings)]

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string."""
        return self.embed([query])[0]
