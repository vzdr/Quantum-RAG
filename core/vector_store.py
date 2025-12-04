"""
Vector Store Module - Phase A, Step 4

Provides ChromaDB-based vector storage with:
- Persistent storage
- Cosine similarity search
- Metadata filtering
"""
import os
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

import chromadb
from chromadb.config import Settings

from .chunker import Chunk
from .embedder import EmbeddedChunk


class VectorStore:
    """
    ChromaDB-based vector store for RAG system.

    Features:
    - Persistent storage
    - Cosine similarity search
    - Metadata storage and filtering
    """

    def __init__(
        self,
        collection_name: str = "rag_collection",
        persist_directory: str = "./chroma_db",
        reset: bool = False
    ):
        """
        Initialize the vector store.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory for persistent storage
            reset: If True, delete existing collection and start fresh
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # Create persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)

        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Delete existing collection if reset is True
        if reset:
            try:
                self.client.delete_collection(collection_name)
                print(f"Deleted existing collection: {collection_name}")
            except Exception:
                pass

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # Store chunk references locally
        self._chunks_map: Dict[str, Chunk] = {}

    @property
    def count(self) -> int:
        """Get the number of items in the collection."""
        return self.collection.count()

    def add(self, embedded_chunks: List[EmbeddedChunk]) -> int:
        """
        Add embedded chunks to the vector store.

        Args:
            embedded_chunks: List of EmbeddedChunk objects

        Returns:
            Number of chunks added
        """
        if not embedded_chunks:
            return 0

        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for ec in embedded_chunks:
            chunk_id = ec.chunk.id

            ids.append(chunk_id)
            embeddings.append(ec.embedding.tolist())
            documents.append(ec.chunk.text)
            metadatas.append({
                "source": ec.chunk.source,
                "chunk_index": ec.chunk.chunk_index,
                "start_char": ec.chunk.start_char,
                "end_char": ec.chunk.end_char,
                **{k: str(v) for k, v in ec.chunk.metadata.items()}
            })

            # Store reference
            self._chunks_map[chunk_id] = ec.chunk

        # Add to ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        return len(ids)

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            where: Optional metadata filter
            where_document: Optional document content filter

        Returns:
            List of search results with id, text, metadata, distance, score
        """
        # Prepare query parameters
        query_params = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": k,
            "include": ["documents", "metadatas", "distances", "embeddings"]
        }

        if where:
            query_params["where"] = where
        if where_document:
            query_params["where_document"] = where_document

        # Execute query
        results = self.collection.query(**query_params)

        # Format results
        output = []
        for i in range(len(results['ids'][0])):
            distance = results['distances'][0][i]
            # Convert cosine distance to similarity score
            # ChromaDB returns cosine distance, so similarity = 1 - distance
            score = 1 - distance

            output.append({
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': distance,
                'score': score,
                'embedding': results['embeddings'][0][i] if results.get('embeddings') else None
            })

        return output

    def get_all(self, include_embeddings: bool = True) -> Dict[str, Any]:
        """
        Get all items from the collection.

        Args:
            include_embeddings: Whether to include embeddings

        Returns:
            Dictionary with ids, documents, metadatas, and optionally embeddings
        """
        include = ["documents", "metadatas"]
        if include_embeddings:
            include.append("embeddings")

        return self.collection.get(include=include)

    def get_all_embeddings(self) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Get all embeddings and metadata for visualization.

        Returns:
            Tuple of (embeddings array, list of metadata dicts)
        """
        results = self.get_all(include_embeddings=True)

        if not results['ids']:
            return np.array([]), []

        embeddings = np.array(results['embeddings'])
        metadata = [
            {
                "id": id_,
                "text": doc,
                "source": meta.get('source', 'unknown'),
                **meta
            }
            for id_, doc, meta in zip(
                results['ids'],
                results['documents'],
                results['metadatas']
            )
        ]

        return embeddings, metadata

    def delete(self, ids: List[str]) -> int:
        """
        Delete chunks by ID.

        Args:
            ids: List of chunk IDs to delete

        Returns:
            Number of chunks deleted
        """
        self.collection.delete(ids=ids)

        # Remove from local map
        for id_ in ids:
            self._chunks_map.pop(id_, None)

        return len(ids)

    def clear(self) -> int:
        """
        Clear all items from the collection.

        Returns:
            Number of items deleted
        """
        count = self.count
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self._chunks_map.clear()
        return count

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary with statistics
        """
        results = self.get_all(include_embeddings=False)

        if not results['ids']:
            return {
                'total_chunks': 0,
                'unique_sources': 0,
                'sources': [],
            }

        sources = list(set(
            meta.get('source', 'unknown')
            for meta in results['metadatas']
        ))

        return {
            'total_chunks': len(results['ids']),
            'unique_sources': len(sources),
            'sources': sources,
            'collection_name': self.collection_name,
            'persist_directory': self.persist_directory,
        }

    def get_by_source(self, source: str) -> List[Dict[str, Any]]:
        """
        Get all chunks from a specific source.

        Args:
            source: Source filename

        Returns:
            List of chunk data
        """
        results = self.collection.get(
            where={"source": source},
            include=["documents", "metadatas"]
        )

        return [
            {
                'id': id_,
                'text': doc,
                'metadata': meta
            }
            for id_, doc, meta in zip(
                results['ids'],
                results['documents'],
                results['metadatas']
            )
        ]
