"""RAG System Configuration."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RAGConfig:
    """
    Global configuration for the RAG system.

    Attributes:
        chunk_size: Number of characters per chunk (100-2000)
        chunk_overlap: Number of overlapping characters between chunks (0-200)
        chunking_strategy: Strategy for chunking ('fixed', 'sentence', 'paragraph')
        embedding_model: Name of the sentence-transformers model
        embedding_batch_size: Batch size for embedding generation
        embedding_device: Device for embeddings ('cpu' or 'cuda')
        collection_name: Name of the ChromaDB collection
        persist_directory: Directory to persist ChromaDB data
        top_k: Number of chunks to retrieve
        similarity_threshold: Minimum similarity score threshold
        llm_model: Gemini model name
        temperature: LLM temperature (0.0-2.0)
        max_tokens: Maximum tokens in LLM response
        system_prompt: System prompt for the LLM
    """

    # Chunking parameters
    chunk_size: int = 500
    chunk_overlap: int = 50
    chunking_strategy: str = "sentence"  # fixed, sentence, paragraph

    # Embedding parameters
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = 32
    embedding_device: str = "cpu"

    # Vector store parameters
    collection_name: str = "rag_collection"
    persist_directory: str = "./chroma_db"

    # Retrieval parameters
    top_k: int = 5
    similarity_threshold: float = 0.0

    # Generation parameters
    llm_model: str = "gemini-2.5-flash-lite"
    temperature: float = 0.7
    max_tokens: int = 1024

    # System prompt
    system_prompt: str = field(default="""You are a helpful assistant that answers questions based on the provided context.
If the answer cannot be found in the context, say so clearly.
Always be accurate and cite the relevant parts of the context in your answer.""")

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'chunking_strategy': self.chunking_strategy,
            'embedding_model': self.embedding_model,
            'embedding_batch_size': self.embedding_batch_size,
            'embedding_device': self.embedding_device,
            'collection_name': self.collection_name,
            'persist_directory': self.persist_directory,
            'top_k': self.top_k,
            'similarity_threshold': self.similarity_threshold,
            'llm_model': self.llm_model,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'system_prompt': self.system_prompt,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'RAGConfig':
        """Create config from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
