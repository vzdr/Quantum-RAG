"""
Retrieval Strategies Module

Implements strategy pattern for diverse retrieval methods:
- NaiveRetrievalStrategy: Top-k by similarity (baseline)
- MMRRetrievalStrategy: Maximal Marginal Relevance (Carbonell & Goldstein, 1998)
- QUBORetrievalStrategy: QUBO-based diverse retrieval with ORBIT

Usage:
    strategy = create_retrieval_strategy('mmr', lambda_param=0.5)
    results, metadata = strategy.retrieve(query_embedding, candidates, k=10)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from .chunker import Chunk


class RetrievalResult:
    """Represents a single retrieval result."""
    def __init__(self, chunk: Chunk, score: float, rank: int):
        self.chunk = chunk
        self.score = score
        self.rank = rank

    @property
    def id(self) -> str:
        return self.chunk.id

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def source(self) -> str:
        return self.chunk.source


class AbstractRetrievalStrategy(ABC):
    """Abstract base class for retrieval strategies."""

    @abstractmethod
    def retrieve(
        self,
        query_embedding: np.ndarray,
        candidate_results: List[Dict[str, Any]],
        k: int,
        threshold: float = 0.0
    ) -> Tuple[List[RetrievalResult], Dict[str, Any]]:
        """
        Retrieve k diverse/relevant results.

        Args:
            query_embedding: Query embedding vector
            candidate_results: Raw search results from vector store
            k: Number of results to select
            threshold: Minimum similarity score threshold

        Returns:
            (retrieval_results, metadata_dict)
            - retrieval_results: List of RetrievalResult objects
            - metadata: Strategy-specific metrics
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return strategy name."""
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Return current strategy parameters."""
        pass


class NaiveRetrievalStrategy(AbstractRetrievalStrategy):
    """Baseline: Top-k by cosine similarity."""

    def get_name(self) -> str:
        return "naive"

    def get_parameters(self) -> Dict[str, Any]:
        return {}

    def retrieve(
        self,
        query_embedding: np.ndarray,
        candidate_results: List[Dict[str, Any]],
        k: int,
        threshold: float = 0.0
    ) -> Tuple[List[RetrievalResult], Dict[str, Any]]:
        """Simple top-k by similarity score."""

        # Filter by threshold
        filtered = [r for r in candidate_results if r['score'] >= threshold]

        # Take top k
        top_k = filtered[:k]

        # Convert to RetrievalResult objects
        retrieval_results = self._convert_to_results(top_k)

        metadata = {
            'method': 'naive',
            'candidates_considered': len(candidate_results),
            'threshold_filtered': len(filtered)
        }

        return retrieval_results, metadata

    def _convert_to_results(self, results: List[Dict[str, Any]]) -> List[RetrievalResult]:
        """Convert raw results to RetrievalResult objects."""
        retrieval_results = []
        for rank, result in enumerate(results, start=1):
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


class MMRRetrievalStrategy(AbstractRetrievalStrategy):
    """
    Maximal Marginal Relevance (Carbonell & Goldstein, 1998).

    MMR = λ * Sim(q, d) - (1-λ) * max[Sim(d, dⱼ) for dⱼ in Selected]

    Greedy iterative selection maximizing MMR score.
    """

    def __init__(self, lambda_param: float = 0.5):
        """
        Initialize MMR strategy.

        Args:
            lambda_param: Relevance-diversity tradeoff (0=diversity, 1=relevance)
                         Standard value: 0.5
        """
        self.lambda_param = lambda_param

    def get_name(self) -> str:
        return "mmr"

    def get_parameters(self) -> Dict[str, Any]:
        return {'lambda': self.lambda_param}

    def retrieve(
        self,
        query_embedding: np.ndarray,
        candidate_results: List[Dict[str, Any]],
        k: int,
        threshold: float = 0.0
    ) -> Tuple[List[RetrievalResult], Dict[str, Any]]:
        """
        Maximal Marginal Relevance selection.

        MMR = λ * Sim(q, d) - (1-λ) * max[Sim(d, dⱼ) for dⱼ in Selected]
        """

        # Filter by threshold
        candidates = [r for r in candidate_results if r['score'] >= threshold]
        if len(candidates) <= k:
            return self._convert_to_results(candidates[:k]), {'method': 'mmr', 'lambda': self.lambda_param}

        # Extract embeddings
        candidate_embeddings = np.array([r['embedding'] for r in candidates])
        query_sims = np.array([r['score'] for r in candidates])

        # Normalize embeddings for pairwise similarity
        candidate_norms = candidate_embeddings / np.linalg.norm(
            candidate_embeddings, axis=1, keepdims=True
        )

        selected_indices = []
        remaining_indices = list(range(len(candidates)))

        # Iteratively select k documents
        for _ in range(min(k, len(candidates))):
            if not remaining_indices:
                break

            mmr_scores = []
            for idx in remaining_indices:
                # Relevance term
                relevance = query_sims[idx]

                # Diversity term (max similarity to already selected)
                if selected_indices:
                    similarities_to_selected = np.dot(
                        candidate_norms[selected_indices],
                        candidate_norms[idx]
                    )
                    max_sim_to_selected = np.max(similarities_to_selected)
                else:
                    max_sim_to_selected = 0.0

                # MMR score
                mmr = self.lambda_param * relevance - (1 - self.lambda_param) * max_sim_to_selected
                mmr_scores.append(mmr)

            # Select document with highest MMR
            best_idx = remaining_indices[np.argmax(mmr_scores)]
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        # Convert to results
        selected_results = [candidates[i] for i in selected_indices]

        metadata = {
            'method': 'mmr',
            'lambda': self.lambda_param,
            'candidates_considered': len(candidates),
            'iterations': len(selected_indices)
        }

        return self._convert_to_results(selected_results), metadata

    def _convert_to_results(self, results: List[Dict[str, Any]]) -> List[RetrievalResult]:
        """Convert raw results to RetrievalResult objects."""
        retrieval_results = []
        for rank, result in enumerate(results, start=1):
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


# ORBIT Solver Configuration Presets
# Trade-off between speed and solution quality
SOLVER_PRESETS = {
    'fast': {
        'n_replicas': 2,
        'full_sweeps': 5000,
        'beta_initial': 0.35,
        'beta_end': 3.5,
        'beta_step_interval': 1
    },
    'balanced': {
        'n_replicas': 4,
        'full_sweeps': 10000,
        'beta_initial': 0.35,
        'beta_end': 3.5,
        'beta_step_interval': 1
    },
    'quality': {
        'n_replicas': 6,
        'full_sweeps': 12000,
        'beta_initial': 0.2,  # More exploration
        'beta_end': 4.0,      # Stronger convergence
        'beta_step_interval': 2  # Slower annealing
    },
    'maximum': {
        'n_replicas': 8,
        'full_sweeps': 15000,
        'beta_initial': 0.15,  # Maximum exploration
        'beta_end': 4.5,       # Maximum convergence
        'beta_step_interval': 2
    }
}


class QUBORetrievalStrategy(AbstractRetrievalStrategy):
    """QUBO-based diverse retrieval using a selectable backend solver."""

    def __init__(
        self,
        alpha: float = 0.05,
        penalty: float = 1000.0,
        solver: str = 'orbit',
        solver_options: Optional[Dict[str, Any]] = None,
        solver_preset: str = 'balanced'
    ):
        """
        Initialize QUBO strategy.

        Args:
            alpha: Diversity weight (default: 0.05). Controls penalty for similar documents.
            penalty: Cardinality constraint penalty (default: 1000.0). Enforces exactly K selections.
            solver: Backend solver ('orbit', 'gurobi', 'bruteforce').
            solver_options: Solver-specific parameters (overrides preset if provided).
            solver_preset: For 'orbit' solver, preset configuration ('fast', 'balanced', 'quality').
        """
        self.alpha = alpha
        self.penalty = penalty
        self.solver = solver
        self.solver_preset = solver_preset

        # Process solver options and presets
        if solver_options is not None:
            self.solver_options = solver_options
            self.solver_preset = 'custom'
        elif self.solver == 'orbit':
            # Store preset name, will be used in solve_diverse_retrieval_qubo
            self.solver_options = {'preset': solver_preset}
        else:
            # For non-orbit solvers, default to empty options if none provided
            self.solver_options = {}


    def get_name(self) -> str:
        return f"qubo_{self.solver}"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            'alpha': self.alpha,
            'penalty': self.penalty,
            'solver': self.solver,
            'solver_preset': self.solver_preset,
            **self.solver_options
        }

    def retrieve(
        self,
        query_embedding: np.ndarray,
        candidate_results: List[Dict[str, Any]],
        k: int,
        threshold: float = 0.0
    ) -> Tuple[List[RetrievalResult], Dict[str, Any]]:
        """
        QUBO-based diverse retrieval.

        minimize: -Σᵢ sim(q, cᵢ) * xᵢ + α * Σᵢⱼ sim(cᵢ, cⱼ) * xᵢ * xⱼ + P * (Σxᵢ - k)²
        subject to: Σxᵢ = k
        """
        from .qubo_solver import solve_diverse_retrieval_qubo

        # Filter by threshold
        candidates = [r for r in candidate_results if r['score'] >= threshold]
        if len(candidates) <= k:
            return self._convert_to_results(candidates[:k]), {'method': 'qubo', 'alpha': self.alpha}

        # Extract embeddings
        candidate_embeddings = np.array([r['embedding'] for r in candidates])

        # Solve QUBO
        selected_indices, solver_metadata = solve_diverse_retrieval_qubo(
            query_embedding=query_embedding,
            candidate_embeddings=candidate_embeddings,
            k=k,
            alpha=self.alpha,
            penalty=self.penalty,
            solver=self.solver,
            solver_options=self.solver_options
        )

        # Convert to results
        selected_results = [candidates[i] for i in selected_indices]

        metadata = {
            'method': 'qubo',
            'candidates_considered': len(candidates),
            **solver_metadata
        }

        return self._convert_to_results(selected_results), metadata

    def _convert_to_results(self, results: List[Dict[str, Any]]) -> List[RetrievalResult]:
        """Convert raw results to RetrievalResult objects."""
        retrieval_results = []
        for rank, result in enumerate(results, start=1):
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


def create_retrieval_strategy(method: str, **params) -> AbstractRetrievalStrategy:
    """
    Factory function to create retrieval strategy.

    Args:
        method: Strategy name ('naive', 'mmr', 'qubo')
        **params: Strategy-specific parameters

    Returns:
        Configured strategy instance

    Examples:
        strategy = create_retrieval_strategy('naive')
        strategy = create_retrieval_strategy('mmr', lambda_param=0.7)
        strategy = create_retrieval_strategy('qubo', alpha=0.65, solver_params={...})
    """
    strategies = {
        'naive': NaiveRetrievalStrategy,
        'mmr': MMRRetrievalStrategy,
        'qubo': QUBORetrievalStrategy
    }

    if method not in strategies:
        raise ValueError(f"Unknown retrieval method: {method}. Choose from {list(strategies.keys())}")

    return strategies[method](**params)
