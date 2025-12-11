"""
Vector Store Module using ChromaDB.
"""
import os
from typing import List, Dict, Any
import numpy as np
import chromadb
from .data_models import EmbeddedChunk

class VectorStore:
    """A simplified ChromaDB-based vector store."""
    def __init__(self, collection_name: str = "rag_collection", persist_directory: str = "./data/vector_dbs", reset: bool = False):
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        if reset:
            try:
                self.client.delete_collection(collection_name)
            except Exception:
                pass
        self.collection = self.client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    def add(self, embedded_chunks: List[EmbeddedChunk], batch_size: int = 500):
        """Adds embedded chunks to the store in batches."""
        if not embedded_chunks:
            return
        for i in range(0, len(embedded_chunks), batch_size):
            batch = embedded_chunks[i:i + batch_size]
            self.collection.add(
                ids=[ec.chunk.id for ec in batch],
                embeddings=[ec.embedding.tolist() for ec in batch],
                documents=[ec.chunk.text for ec in batch],
                metadatas=[{**ec.chunk.metadata, "source": ec.chunk.source} for ec in batch]
            )

    def search(self, query_embedding: np.ndarray, k: int = 5, where: Dict = None) -> List[Dict[str, Any]]:
        """Searches for similar chunks."""
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances", "embeddings"]
        )
        
        output = []
        if not results['ids'][0]:
            return output

        for i in range(len(results['ids'][0])):
            output.append({
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'score': 1 - results['distances'][0][i],
                'embedding': np.array(results['embeddings'][0][i])
            })
        return output

    def get_all_items(self):
        """Retrieves all items from the collection."""
        return self.collection.get(include=["metadatas", "documents"])

    def get_all_embeddings(self):
        """Retrieves all embeddings and metadata."""
        results = self.collection.get(include=["embeddings", "metadatas"])
        embeddings = np.array(results['embeddings']) if len(results['embeddings']) > 0 else np.array([])
        metadata = [{'id': id_, **meta} for id_, meta in zip(results['ids'], results['metadatas'])]
        return embeddings, metadata

    def add_with_embeddings(self, chunks: List[Dict[str, Any]], batch_size: int = 500):
        """Add chunks that already have embeddings."""
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            self.collection.add(
                ids=[c['id'] for c in batch],
                embeddings=[c['embedding'].tolist() for c in batch],
                documents=[c['text'] for c in batch],
                metadatas=[c['metadata'] for c in batch]
            )

    @property
    def count(self) -> int:
        """Get number of items in collection."""
        return self.collection.count()

    def clear(self):
        """Clear all items from collection."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(name=self.collection.name, metadata={"hnsw:space": "cosine"})

    def get_statistics(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return {"count": self.count, "name": self.collection.name}
