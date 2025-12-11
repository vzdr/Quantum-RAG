"""
Core data models for the RAG system.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any
import numpy as np

@dataclass
class Document:
    """Represents a loaded document."""
    content: str
    source: str
    file_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Chunk:
    """Represents a text chunk."""
    id: str
    text: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EmbeddedChunk:
    """Represents a chunk with its embedding."""
    chunk: Chunk
    embedding: np.ndarray

@dataclass
class RetrievalResult:
    """Represents a single retrieval result."""
    chunk: Chunk
    score: float
    rank: int

@dataclass
class GenerationResult:
    """Represents a generation result."""
    query: str
    response: str
    context_chunks: List[RetrievalResult]
    model: str
