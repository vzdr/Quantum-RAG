"""
Retriever Module - Phase B, Steps 5-7

Handles:
- Query embedding (Step 5)
- Similarity search (Step 6)
- Top-K selection (Step 7) with configurable retrieval strategies
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np

from .chunker import Chunk
from .embedder import EmbeddingGenerator
from .vector_store import VectorStore
from .retrieval_strategies import (
    AbstractRetrievalStrategy,
    create_retrieval_strategy,
    RetrievalResult as StrategyRetrievalResult
)


@dataclass
class RetrievalResult:
    """Represents a retrieval result."""
    chunk: Chunk
    score: float
    rank: int

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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'text': self.text,
            'source': self.source,
            'score': self.score,
            'rank': self.rank,
            'chunk_index': self.chunk.chunk_index,
        }


class Retriever:
    """
    Handles query processing and retrieval.

    Implements:
    - Step 5: Query embedding
    - Step 6: Similarity search
    - Step 7: Top-K selection
    """

    def __init__(
        self,
        embedder: EmbeddingGenerator,
        vector_store: VectorStore,
        retrieval_strategy: Optional[AbstractRetrievalStrategy] = None,
        retrieval_method: str = 'naive',
        strategy_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the retriever.

        Args:
            embedder: EmbeddingGenerator for query embedding
            vector_store: VectorStore for similarity search
            retrieval_strategy: Pre-configured strategy (optional)
            retrieval_method: Strategy name if creating new ('naive', 'mmr', 'qubo')
            strategy_params: Parameters for strategy creation
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self._last_query_embedding: Optional[np.ndarray] = None
        self._last_retrieval_metadata: Optional[Dict[str, Any]] = None

        # Initialize strategy
        if retrieval_strategy is not None:
            self.strategy = retrieval_strategy
        else:
            params = strategy_params or {}
            self.strategy = create_retrieval_strategy(retrieval_method, **params)

    @property
    def last_query_embedding(self) -> Optional[np.ndarray]:
        """Get the last query embedding for visualization."""
        return self._last_query_embedding

    @property
    def last_retrieval_metadata(self) -> Optional[Dict[str, Any]]:
        """Get metadata from last retrieval."""
        return self._last_retrieval_metadata

    def embed_query(self, query: str) -> np.ndarray:
        """
        Step 5: Create vector embedding for the query.

        Args:
            query: Query text

        Returns:
            Query embedding vector
        """
        self._last_query_embedding = self.embedder.embed_query(query)
        return self._last_query_embedding

    def similarity_search(
        self,
        query_embedding: np.ndarray,
        k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Step 6: Find cosine similarity between query and chunk vectors.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return

        Returns:
            List of search results with similarity scores
        """
        return self.vector_store.search(query_embedding, k=k)

    def select_top_k(
        self,
        results: List[Dict[str, Any]],
        k: int = 5,
        threshold: float = 0.0
    ) -> List[RetrievalResult]:
        """
        Step 7: Select top-k chunks based on similarity scores.

        Args:
            results: Raw search results
            k: Number of top results to select
            threshold: Minimum similarity score threshold

        Returns:
            List of RetrievalResult objects
        """
        # Filter by threshold
        filtered = [r for r in results if r['score'] >= threshold]

        # Take top k
        top_k = filtered[:k]

        # Convert to RetrievalResult objects
        retrieval_results = []
        for rank, result in enumerate(top_k, start=1):
            chunk = Chunk(
                id=result['id'],
                text=result['text'],
                source=result['metadata'].get('source', 'unknown'),
                chunk_index=int(result['metadata'].get('chunk_index', 0)),
                start_char=int(result['metadata'].get('start_char', 0)),
                end_char=int(result['metadata'].get('end_char', 0)),
                metadata={k: v for k, v in result['metadata'].items()
                         if k not in ['source', 'chunk_index', 'start_char', 'end_char']}
            )
            retrieval_results.append(RetrievalResult(
                chunk=chunk,
                score=result['score'],
                rank=rank
            ))

        return retrieval_results

    def retrieve(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.0
    ) -> List[RetrievalResult]:
        """
        Complete retrieval pipeline: embed query -> search -> strategy selection.

        Args:
            query: Query text
            k: Number of results to return
            threshold: Minimum similarity score threshold

        Returns:
            List of RetrievalResult objects
        """
        # Step 5: Embed query
        query_embedding = self.embed_query(query)

        # Step 6: Similarity search (get more candidates for diversity strategies)
        search_factor = 3 if self.strategy.get_name() in ['mmr', 'qubo'] else 2
        search_results = self.similarity_search(query_embedding, k=k * search_factor)

        # Step 7: Apply retrieval strategy
        results, metadata = self.strategy.retrieve(
            query_embedding=query_embedding,
            candidate_results=search_results,
            k=k,
            threshold=threshold
        )

        # Store metadata
        self._last_retrieval_metadata = metadata

        # Convert strategy results to retriever results
        return self._convert_strategy_results(results)

    def _convert_strategy_results(self, strategy_results: List[StrategyRetrievalResult]) -> List[RetrievalResult]:
        """Convert strategy results to retriever results."""
        return [
            RetrievalResult(chunk=r.chunk, score=r.score, rank=r.rank)
            for r in strategy_results
        ]

    def set_strategy(self, method: str, **params):
        """
        Change retrieval strategy dynamically.

        Args:
            method: Strategy name ('naive', 'mmr', 'qubo')
            **params: Strategy-specific parameters
        """
        self.strategy = create_retrieval_strategy(method, **params)

    def get_retrieval_statistics(
        self,
        results: List[RetrievalResult]
    ) -> Dict[str, Any]:
        """
        Get statistics about retrieval results.

        Args:
            results: List of retrieval results

        Returns:
            Dictionary with statistics
        """
        if not results:
            return {
                'num_results': 0,
                'avg_score': 0,
                'max_score': 0,
                'min_score': 0,
                'unique_sources': 0,
            }

        scores = [r.score for r in results]
        sources = list(set(r.source for r in results))

        return {
            'num_results': len(results),
            'avg_score': sum(scores) / len(scores),
            'max_score': max(scores),
            'min_score': min(scores),
            'unique_sources': len(sources),
            'sources': sources,
        }

    def format_context(
        self,
        results: List[RetrievalResult],
        include_scores: bool = True,
        include_sources: bool = True
    ) -> str:
        """
        Format retrieval results as context for LLM.

        Args:
            results: List of retrieval results
            include_scores: Whether to include similarity scores
            include_sources: Whether to include source information

        Returns:
            Formatted context string
        """
        context_parts = []

        for result in results:
            header_parts = []
            if include_sources:
                header_parts.append(f"Source: {result.source}")
            if include_scores:
                header_parts.append(f"Score: {result.score:.3f}")

            if header_parts:
                header = f"[{', '.join(header_parts)}]"
                context_parts.append(f"{header}\n{result.text}")
            else:
                context_parts.append(result.text)

        return "\n\n---\n\n".join(context_parts)

    def get_retrieved_indices(
        self,
        results: List[RetrievalResult],
        all_metadata: List[Dict[str, Any]]
    ) -> List[int]:
        """
        Get indices of retrieved chunks in the full embedding list.

        Useful for visualization highlighting.

        Args:
            results: List of retrieval results
            all_metadata: List of all chunk metadata from vector store

        Returns:
            List of indices (in same order as results)
        """
        # Build a map from id to index in all_metadata
        id_to_index = {meta.get('id'): i for i, meta in enumerate(all_metadata)}

        # Return indices in the same order as results (preserves score order)
        indices = []
        for r in results:
            if r.id in id_to_index:
                indices.append(id_to_index[r.id])

        return indices
