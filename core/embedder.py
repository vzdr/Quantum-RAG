"""
Embedding Generator Module - Phase A, Step 3

Generates embeddings using sentence-transformers.
Features:
- Batch processing with progress tracking
- Multiple model support
- CPU/GPU device selection
"""
import torch
from dataclasses import dataclass
from typing import List, Optional, Callable
import numpy as np

from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from .chunker import Chunk


@dataclass
class EmbeddedChunk:
    """Represents a chunk with its embedding."""
    chunk: Chunk
    embedding: np.ndarray

    @property
    def id(self) -> str:
        """Return chunk ID."""
        return self.chunk.id

    @property
    def text(self) -> str:
        """Return chunk text."""
        return self.chunk.text

    @property
    def source(self) -> str:
        """Return chunk source."""
        return self.chunk.source


class EmbeddingGenerator:
    """
    Generates embeddings using sentence-transformers.

    Supported models include:
    - all-MiniLM-L6-v2 (fast, 384 dimensions)
    - all-mpnet-base-v2 (balanced, 768 dimensions)
    - multi-qa-mpnet-base-cos-v1 (QA optimized)
    - paraphrase-multilingual-MiniLM-L12-v2 (multilingual)
    """

    AVAILABLE_MODELS = {
        'all-MiniLM-L6-v2': {
            'dimensions': 384,
            'description': 'Fast, good quality (384d)',
        },
        'all-mpnet-base-v2': {
            'dimensions': 768,
            'description': 'Balanced performance (768d)',
        },
        'multi-qa-mpnet-base-cos-v1': {
            'dimensions': 768,
            'description': 'Optimized for QA tasks (768d)',
        },
        'paraphrase-multilingual-MiniLM-L12-v2': {
            'dimensions': 384,
            'description': 'Multilingual support (384d)',
        },
        'BAAI/bge-large-en-v1.5': {
            'dimensions': 1024,
            'description': 'High-quality embeddings for retrieval (1024d)',
        },
    }

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = None
    ):
        """
        Initialize the embedding generator.

        Args:
            model_name: Name of the sentence-transformers model
            device: Device to use ('cpu' or 'cuda'). If None, will auto-detect.
        """
        self.model_name = model_name
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self._model: Optional[SentenceTransformer] = None
        self._embedding_dim: Optional[int] = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the model."""
        if self._model is None:
            print(f"Loading embedding model: {self.model_name}...")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._embedding_dim = self._model.get_sentence_embedding_dimension()
            print(f"Model loaded (dim={self._embedding_dim})")
        return self._model

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        if self._embedding_dim is None:
            _ = self.model  # Trigger model loading
        return self._embedding_dim

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text string.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array
        """
        return self.model.encode(text, convert_to_numpy=True)

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Embed multiple texts.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar

        Returns:
            Array of embeddings (n_texts, embedding_dim)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        return embeddings

    def embed_chunks(
        self,
        chunks: List[Chunk],
        batch_size: int = 32,
        show_progress: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[EmbeddedChunk]:
        """
        Embed a list of chunks.

        Args:
            chunks: List of Chunk objects
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            List of EmbeddedChunk objects
        """
        texts = [chunk.text for chunk in chunks]

        if show_progress and progress_callback is None:
            # Use tqdm progress bar
            embeddings = []
            for i in tqdm(range(0, len(texts), batch_size), desc="Embedding chunks"):
                batch = texts[i:i + batch_size]
                batch_embeddings = self.model.encode(
                    batch,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                embeddings.extend(batch_embeddings)
            embeddings = np.array(embeddings)
        elif progress_callback is not None:
            # Use custom callback
            embeddings = []
            total_batches = (len(texts) + batch_size - 1) // batch_size
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self.model.encode(
                    batch,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                embeddings.extend(batch_embeddings)
                current_batch = (i // batch_size) + 1
                progress_callback(current_batch, total_batches)
            embeddings = np.array(embeddings)
        else:
            # No progress tracking
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False
            )

        # Create EmbeddedChunk objects
        embedded_chunks = [
            EmbeddedChunk(chunk=chunk, embedding=emb)
            for chunk, emb in zip(chunks, embeddings)
        ]

        return embedded_chunks

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a query string.

        Args:
            query: Query text

        Returns:
            Query embedding as numpy array
        """
        return self.embed_text(query)

    def compute_similarity(
        self,
        query_embedding: np.ndarray,
        chunk_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and chunks.

        Args:
            query_embedding: Query embedding vector (embedding_dim,)
            chunk_embeddings: Chunk embeddings matrix (n_chunks, embedding_dim)

        Returns:
            Similarity scores (n_chunks,)
        """
        # Normalize vectors
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        chunk_norms = chunk_embeddings / np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)

        # Compute cosine similarity
        similarities = np.dot(chunk_norms, query_norm)

        return similarities

    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'device': self.device,
            'max_seq_length': self.model.max_seq_length,
        }
