"""
Retrieval Service - Wraps existing core modules for the demo API.
"""

import sys
import time
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Add core to path
CORE_PATH = Path(__file__).parent.parent.parent.parent / "core"
sys.path.insert(0, str(CORE_PATH.parent))

from core.embedder import EmbeddingGenerator
from core.vector_store import VectorStore
from core.document_loader import DocumentLoader
from core.chunker import TextChunker
from core.retrieval_strategies import (
    create_retrieval_strategy,
    NaiveRetrievalStrategy,
    MMRRetrievalStrategy,
    QUBORetrievalStrategy,
)
from core.diversity_metrics import (
    compute_intra_list_similarity,
    compute_cluster_coverage_from_filenames,
)
from core.generator import ResponseGenerator

from ..models.schemas import (
    RetrievalResult,
    RetrievalMetrics,
    MethodResult,
)


class RetrievalService:
    """
    Service for running retrieval comparisons across multiple methods.
    """

    # Dataset configurations
    DATASETS = {
        "medical": {
            "path": "data/medical/raw",
            "collection": "medical_conditions_demo",
            "total_clusters": 22,  # 22 medical conditions
        },
        "greedy_trap": {
            "path": "data/greedy_trap",
            "collection": "greedy_trap_demo",
            "total_clusters": 7,  # 7 symptom clusters
        },
        "legal": {
            "path": "data/legal_cases",
            "collection": "legal_demo",
            "total_clusters": 5,  # 5 legal categories
        },
        "wikipedia": {
            "path": "data/wikipedia",
            "collection": "wikipedia_general",
            "total_clusters": 171,  # 171 Wikipedia articles
            "loader": "wikipedia",  # Use Wikipedia JSONL+NPZ loader
        },
    }

    def __init__(
        self,
        embedding_model: str = "BAAI/bge-large-en-v1.5",  # 1024-dim to match Wikipedia dataset
        device: str = "cpu",
        persist_dir: str = "./demo_chroma_db",
    ):
        """
        Initialize the retrieval service.

        Args:
            embedding_model: Sentence transformer model name
            device: Device for embeddings (cpu/cuda)
            persist_dir: Directory for ChromaDB persistence
        """
        self.embedding_model = embedding_model
        self.device = device
        self.persist_dir = persist_dir

        # Lazy-loaded components
        self._embedder: Optional[EmbeddingGenerator] = None
        self._vector_stores: Dict[str, VectorStore] = {}
        self._generator: Optional[ResponseGenerator] = None

        # Thread pool for parallel execution
        self._executor = ThreadPoolExecutor(max_workers=3)

    @property
    def embedder(self) -> EmbeddingGenerator:
        """Lazy load embedder."""
        if self._embedder is None:
            self._embedder = EmbeddingGenerator(
                model_name=self.embedding_model,
                device=self.device
            )
        return self._embedder

    @property
    def generator(self) -> ResponseGenerator:
        """Lazy load generator."""
        if self._generator is None:
            self._generator = ResponseGenerator(
                model="gemini-2.5-flash-lite",
                temperature=0.7,
                max_tokens=1024
            )
        return self._generator

    def get_vector_store(self, dataset: str) -> VectorStore:
        """Get or create vector store for dataset."""
        if dataset not in self._vector_stores:
            config = self.DATASETS.get(dataset)
            if not config:
                raise ValueError(f"Unknown dataset: {dataset}")

            self._vector_stores[dataset] = VectorStore(
                collection_name=config["collection"],
                persist_directory=self.persist_dir,
                reset=False
            )
        return self._vector_stores[dataset]

    def index_dataset(self, dataset: str, force_reindex: bool = False) -> int:
        """
        Index a dataset into the vector store.
        Supports both standard document loading and Wikipedia JSONL+NPZ format.

        Args:
            dataset: Dataset name
            force_reindex: Force re-indexing even if data exists

        Returns:
            Number of documents indexed
        """
        config = self.DATASETS.get(dataset)
        if not config:
            raise ValueError(f"Unknown dataset: {dataset}")

        vector_store = self.get_vector_store(dataset)

        # Check if already indexed
        if vector_store.count > 0 and not force_reindex:
            return vector_store.count

        # Clear if reindexing
        if force_reindex:
            vector_store.clear()

        # Load and index documents
        data_path = Path(__file__).parent.parent.parent.parent / config["path"]
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {data_path}")

        # Use Wikipedia loader for special format
        if config.get("loader") == "wikipedia":
            from .dataset_loaders import (
                load_wikipedia_dataset,
                convert_wikipedia_to_chroma_format
            )

            print(f"Loading Wikipedia dataset from {data_path}...")
            chunks, embeddings = load_wikipedia_dataset(data_path)
            chroma_chunks = convert_wikipedia_to_chroma_format(chunks, embeddings)

            # Add with pre-computed embeddings
            vector_store.add_with_embeddings(chroma_chunks)
            print(f"Indexed {vector_store.count} Wikipedia chunks")
        else:
            # Standard document loading pipeline
            documents = DocumentLoader.load_directory(str(data_path))
            chunker = TextChunker(chunk_size=500, overlap=50, strategy='sentence')
            chunks = chunker.chunk_documents(documents)
            embedded_chunks = self.embedder.embed_chunks(chunks)
            vector_store.add(embedded_chunks)

        return vector_store.count

    def _run_single_method(
        self,
        method: str,
        query_embedding: np.ndarray,
        candidates: List[Dict[str, Any]],
        k: int,
        total_clusters: int,
        # Configurable parameters - alpha increased for better diversity demonstration
        alpha: float = 0.15,
        penalty: float = 1000.0,
        lambda_param: float = 0.5,
        solver_preset: str = "balanced",
    ) -> Tuple[str, MethodResult]:
        """Run a single retrieval method and compute metrics."""
        start_time = time.perf_counter()

        # Create strategy with user-provided parameters
        if method == "topk":
            strategy = create_retrieval_strategy("naive")
        elif method == "mmr":
            strategy = create_retrieval_strategy("mmr", lambda_param=lambda_param)
        elif method == "qubo":
            strategy = create_retrieval_strategy(
                "qubo",
                alpha=alpha,
                penalty=penalty,
                solver_preset=solver_preset
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        # Run retrieval
        results, metadata = strategy.retrieve(query_embedding, candidates, k)

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Convert results to schema format
        retrieval_results = [
            RetrievalResult(
                rank=r.rank,
                score=r.score,
                text=r.text,
                source=r.source,
                chunk_id=r.id,
            )
            for r in results
        ]

        # Compute diversity metrics
        # Build results dict for metrics computation
        results_with_embeddings = []
        for r in results:
            # Find the embedding from candidates
            for c in candidates:
                if c['id'] == r.id:
                    results_with_embeddings.append({
                        'id': r.id,
                        'source': r.source,
                        'score': r.score,
                        'embedding': c['embedding'],
                    })
                    break

        try:
            ils = compute_intra_list_similarity(results_with_embeddings)
        except Exception:
            ils = 0.0

        # Compute cluster coverage
        cluster_info = compute_cluster_coverage_from_filenames(
            results_with_embeddings,
            total_clusters=total_clusters,
        )

        # Calculate average relevance
        avg_relevance = np.mean([r.score for r in results]) if results else 0.0

        metrics = RetrievalMetrics(
            latency_ms=latency_ms,
            intra_list_similarity=ils,
            cluster_coverage=cluster_info['coverage_count'],
            total_clusters=total_clusters,
            avg_relevance=float(avg_relevance),
        )

        return method, MethodResult(
            method=method,
            results=retrieval_results,
            metrics=metrics,
            llm_response=None,
        )

    async def compare_methods(
        self,
        query: str,
        dataset: str,
        k: int = 5,
        include_llm: bool = True,
        # Configurable parameters - alpha increased for better diversity
        alpha: float = 0.15,
        penalty: float = 1000.0,
        lambda_param: float = 0.5,
        solver_preset: str = "balanced",
    ) -> Dict[str, Any]:
        """
        Run all three retrieval methods and compare results.
        Now supports configurable parameters for QUBO and MMR.

        Args:
            query: User query
            dataset: Dataset to search
            k: Number of results per method
            include_llm: Whether to generate LLM responses
            alpha: QUBO diversity weight (default: 0.15 for moderate diversity)
            penalty: QUBO cardinality penalty (notebook default: 1000.0)
            lambda_param: MMR lambda parameter (default: 0.5)
            solver_preset: ORBIT solver preset (notebook default: "balanced")

        Returns:
            Dictionary with results from all methods
        """
        # Ensure dataset is indexed
        self.index_dataset(dataset)

        vector_store = self.get_vector_store(dataset)
        config = self.DATASETS[dataset]

        # Embed query
        query_embedding = self.embedder.embed_query(query)

        # Get candidates (more than k for diversity selection)
        num_candidates = min(k * 3, vector_store.count)
        candidates = vector_store.search(query_embedding, k=num_candidates)

        # Run all methods in parallel using thread pool with parameters
        loop = asyncio.get_event_loop()
        tasks = []
        for method in ["topk", "mmr", "qubo"]:
            task = loop.run_in_executor(
                self._executor,
                self._run_single_method,
                method,
                query_embedding,
                candidates,
                k,
                config["total_clusters"],
                alpha,  # Pass parameter
                penalty,  # Pass parameter
                lambda_param,  # Pass parameter
                solver_preset,  # Pass parameter
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        # Convert to dict
        method_results = {method: result for method, result in results}

        # Generate LLM responses if requested
        if include_llm:
            for method, result in method_results.items():
                try:
                    # Convert to format expected by generator
                    from core.retrieval_strategies import RetrievalResult as CoreResult
                    from core.chunker import Chunk

                    core_results = []
                    for r in result.results:
                        chunk = Chunk(
                            id=r.chunk_id,
                            text=r.text,
                            source=r.source,
                            chunk_index=0,
                            start_char=0,
                            end_char=len(r.text),
                        )
                        core_results.append(CoreResult(chunk=chunk, score=r.score, rank=r.rank))

                    gen_result = self.generator.generate(query, core_results)
                    result.llm_response = gen_result.response
                except Exception as e:
                    result.llm_response = f"Error generating response: {str(e)}"

        return {
            "query": query,
            "dataset": dataset,
            "topk": method_results["topk"],
            "mmr": method_results["mmr"],
            "qubo": method_results["qubo"],
        }

    async def retrieve_single(
        self,
        query: str,
        dataset: str,
        method: str,
        k: int = 5,
    ) -> MethodResult:
        """
        Run a single retrieval method.

        Args:
            query: User query
            dataset: Dataset to search
            method: Retrieval method (topk, mmr, qubo)
            k: Number of results

        Returns:
            MethodResult with results and metrics
        """
        # Ensure dataset is indexed
        self.index_dataset(dataset)

        vector_store = self.get_vector_store(dataset)
        config = self.DATASETS[dataset]

        # Embed query
        query_embedding = self.embedder.embed_query(query)

        # Get candidates
        num_candidates = min(k * 3, vector_store.count)
        candidates = vector_store.search(query_embedding, k=num_candidates)

        # Run method
        _, result = self._run_single_method(
            method,
            query_embedding,
            candidates,
            k,
            config["total_clusters"],
        )

        return result

    def get_available_datasets(self) -> List[str]:
        """Get list of available datasets."""
        available = []
        for name, config in self.DATASETS.items():
            data_path = Path(__file__).parent.parent.parent.parent / config["path"]
            if data_path.exists():
                available.append(name)
        return available

    def get_dataset_stats(self, dataset: str) -> Dict[str, Any]:
        """Get statistics for a dataset."""
        vector_store = self.get_vector_store(dataset)
        return vector_store.get_statistics()


# Global service instance
_service_instance: Optional[RetrievalService] = None


def get_retrieval_service() -> RetrievalService:
    """Get or create the global retrieval service instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = RetrievalService()
    return _service_instance
