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

from core.embedding import EmbeddingGenerator
from core.storage import VectorStore
from core.chunking import load_document, chunk_document
from core.retrieval import NaiveRetrieval, MMRRetrieval, QUBORetrieval
from core.generation import ResponseGenerator
from core.utils import (
    compute_intra_list_similarity,
    compute_cluster_coverage_from_filenames,
    compute_aspect_recall,
    filter_chunks_by_prompt,
)
from core.data_models import Chunk

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
                model_name=self.embedding_model
            )
        return self._embedder

    @property
    def generator(self) -> ResponseGenerator:
        """Lazy load generator."""
        if self._generator is None:
            self._generator = ResponseGenerator(
                model="gemini-2.5-flash-lite"
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
            chroma_chunks = convert_wikipedia_to_chroma_format(chunks, embeddings, redundancy_level=4)

            # Add with pre-computed embeddings
            vector_store.add_with_embeddings(chroma_chunks)
            print(f"Indexed {vector_store.count} Wikipedia chunks")
        else:
            # Standard document loading pipeline
            from pathlib import Path as P
            chunks = []
            for file_path in data_path.glob('**/*.txt'):
                doc = load_document(str(file_path))
                doc_chunks = chunk_document(doc, chunk_size=500, overlap=50)
                chunks.extend(doc_chunks)
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
        dataset: str = None,
        # Configurable parameters - production values from experiments
        alpha: float = 0.04,
        beta: float = 0.8,
        penalty: float = 10.0,
        lambda_param: float = 0.85,
        solver_preset: str = "balanced",
    ) -> Tuple[str, MethodResult]:
        """Run a single retrieval method and compute metrics."""
        start_time = time.perf_counter()

        # Create strategy with user-provided parameters
        if method == "topk":
            strategy = NaiveRetrieval()
        elif method == "mmr":
            strategy = MMRRetrieval(lambda_param=lambda_param)
        elif method == "qubo":
            strategy = QUBORetrieval(alpha=alpha, beta=beta, penalty=penalty, solver='gurobi')
        else:
            raise ValueError(f"Unknown method: {method}")

        # Run retrieval
        results = strategy.retrieve(query_embedding, candidates, k)

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Convert results to schema format (include aspect metadata for Wikipedia)
        retrieval_results = [
            RetrievalResult(
                rank=r.rank,
                score=r.score,
                text=r.chunk.text,
                source=r.chunk.source,
                chunk_id=r.chunk.id,
                aspect_id=r.chunk.metadata.get('aspect_id'),
                aspect_name=r.chunk.metadata.get('aspect_name'),
            )
            for r in results
        ]

        # Compute diversity metrics
        # Build results dict for metrics computation
        results_with_embeddings = []
        for r in results:
            # Find the embedding from candidates
            for c in candidates:
                if c['id'] == r.chunk.id:
                    results_with_embeddings.append({
                        'id': r.chunk.id,
                        'source': r.chunk.source,
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

        # Compute aspect recall for Wikipedia dataset
        aspect_recall_pct = None
        aspects_found = None
        total_aspects = 5  # Default for Wikipedia dataset

        if dataset == 'wikipedia':
            # Find the most common article in results
            article_counts = {}
            for r in results:
                article = r.chunk.metadata.get('article_title')
                if article:
                    article_counts[article] = article_counts.get(article, 0) + 1

            # If we have results, compute aspect recall for the most common article
            if article_counts:
                main_article = max(article_counts, key=article_counts.get)

                # Count unique aspects for this article only
                unique_aspects = set()
                for r in results:
                    if r.chunk.metadata.get('article_title') == main_article:
                        aspect_name = r.chunk.metadata.get('aspect_name')
                        chunk_type = r.chunk.metadata.get('chunk_type', '')
                        # Count gold aspects (not noise)
                        if aspect_name and aspect_name not in ['general', 'prompt'] and chunk_type in ['gold_base', 'gold_redundant']:
                            unique_aspects.add(aspect_name)

                aspects_found = len(unique_aspects)
                aspect_recall_pct = (aspects_found / total_aspects) * 100.0 if total_aspects > 0 else 0.0

        metrics = RetrievalMetrics(
            latency_ms=latency_ms,
            intra_list_similarity=ils,
            cluster_coverage=cluster_info['coverage_count'],
            total_clusters=total_clusters,
            avg_relevance=float(avg_relevance),
            aspect_recall=aspect_recall_pct,
            aspects_found=aspects_found,
            total_aspects=total_aspects,
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
        # Configurable parameters - production values from experiments
        alpha: float = 0.04,
        beta: float = 0.8,
        penalty: float = 10.0,
        lambda_param: float = 0.85,
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
            alpha: QUBO diversity weight (default: 0.04, tuned via experiments)
            penalty: QUBO cardinality penalty (default: 10.0, tuned via experiments)
            lambda_param: MMR lambda parameter (default: 0.85, higher = more relevance)
            solver_preset: ORBIT solver preset (default: "balanced")

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
                dataset,  # Pass dataset for aspect recall computation
                alpha,  # Pass parameter
                beta,  # Pass new beta parameter
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
                    from core.data_models import RetrievalResult as CoreResult, Chunk as CoreChunk

                    core_results = []
                    for r in result.results:
                        chunk = CoreChunk(
                            id=r.chunk_id,
                            text=r.text,
                            source=r.source,
                            metadata={}
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
            dataset,
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
