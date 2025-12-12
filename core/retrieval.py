"""
Unified retrieval module with multiple strategies.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from .data_models import Chunk, RetrievalResult
from .embedding import EmbeddingGenerator
from .storage import VectorStore
from .utils import compute_cosine_similarities, compute_pairwise_similarities


def qubo_to_ising(r: np.ndarray, S: np.ndarray, alpha: float, P: float, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert QUBO formulation to Ising Hamiltonian format for ORBIT.

    QUBO Energy: E(x) = -r^T x + alpha * x^T S x + P(sum(x) - k)^2
    where x_i ∈ {0, 1}

    Ising Energy: H(s) = -∑_{i,j} J_{ij} s_i s_j - ∑_i h_i s_i
    where s_i ∈ {-1, +1}

    Conversion: x_i = (s_i + 1) / 2

    Args:
        r: Relevance scores (n,)
        S: Pairwise similarity matrix (n, n)
        alpha: Diversity weight
        P: Penalty coefficient
        k: Target number of selections

    Returns:
        J: Interaction matrix (n, n)
        h: External field vector (n,)
    """
    n = len(r)

    # Quadratic terms (interaction matrix J)
    # After substitution x_i = (s_i+1)/2 into alpha*S_{ij}*x_i*x_j + P*x_i*x_j:
    # Coefficient of s_i*s_j is (alpha*S_{ij} + P) / 4
    J = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            J[i, j] = (alpha * S[i, j] + P) / 4.0
            J[j, i] = J[i, j]  # Symmetric

    # Linear terms (external field h)
    h = np.zeros(n)
    for i in range(n):
        # Relevance: -r_i * (s_i+1)/2 → -r_i/2 * s_i + constant
        h_relevance = -r[i] / 2.0

        # Penalty linear: P(1-2k) * (s_i+1)/2 → P(1-2k)/2 * s_i + constant
        h_penalty = P * (1 - 2*k) / 2.0

        # Diagonal quadratic becomes linear after x_i^2 = x_i substitution
        h_diagonal = (alpha * S[i, i] + P) / 4.0

        h[i] = h_relevance + h_penalty + h_diagonal

    return J, h


def ising_to_qubo_solution(s: np.ndarray) -> np.ndarray:
    """
    Convert Ising spin configuration to QUBO binary variables.

    Args:
        s: Ising spins (n,) with values in {-1, +1}

    Returns:
        x: Binary variables (n,) with values in {0, 1}
    """
    return ((s + 1) / 2).astype(int)


class BaseRetrievalStrategy(ABC):
    """Abstract base class for retrieval strategies."""
    @abstractmethod
    def retrieve(self, query_embedding: np.ndarray, candidates: List[Dict[str, Any]], k: int) -> List[RetrievalResult]:
        pass

class NaiveRetrieval(BaseRetrievalStrategy):
    """Retrieves the top-k most similar candidates."""
    def retrieve(self, query_embedding: np.ndarray, candidates: List[Dict[str, Any]], k: int) -> List[RetrievalResult]:
        # Candidates are pre-sorted by score by the Retriever
        return [RetrievalResult(Chunk(id=r['id'], text=r['text'], source=r['metadata'].get('source', r['metadata'].get('article_title', 'unknown')), metadata=r['metadata']), r['score'], i+1) for i, r in enumerate(candidates[:k])]

class MMRRetrieval(BaseRetrievalStrategy):
    """Maximal Marginal Relevance retrieval."""
    def __init__(self, lambda_param: float = 0.7):
        self.lambda_param = lambda_param

    def retrieve(self, query_embedding: np.ndarray, candidates: List[Dict[str, Any]], k: int) -> List[RetrievalResult]:
        if not candidates: return []
        
        candidate_embeddings = np.array([c['embedding'] for c in candidates])
        query_sims = np.array([c['score'] for c in candidates])
        
        selected_indices = []
        remaining_indices = list(range(len(candidates)))

        # Greedily select the most relevant item first
        best_initial_idx = np.argmax(query_sims)
        selected_indices.append(best_initial_idx)
        remaining_indices.remove(best_initial_idx)

        while len(selected_indices) < k and remaining_indices:
            mmr_scores = []
            for idx in remaining_indices:
                relevance = query_sims[idx]
                
                # Diversity term
                selected_embeds = candidate_embeddings[selected_indices]
                max_sim_to_selected = np.max(compute_cosine_similarities(candidate_embeddings[idx], selected_embeds))
                
                mmr = self.lambda_param * relevance - (1 - self.lambda_param) * max_sim_to_selected
                mmr_scores.append(mmr)

            best_idx_in_remaining = np.argmax(mmr_scores)
            best_idx = remaining_indices[best_idx_in_remaining]
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        return [RetrievalResult(Chunk(id=candidates[i]['id'], text=candidates[i]['text'], source=candidates[i]['metadata'].get('source', candidates[i]['metadata'].get('article_title', 'unknown')), metadata=candidates[i]['metadata']), candidates[i]['score'], rank+1) for rank, i in enumerate(selected_indices)]

class QUBORetrieval(BaseRetrievalStrategy):
    """QUBO-based retrieval using Gurobi or ORBIT solvers."""
    def __init__(self, alpha: float = 0.04, penalty: float = 10.0, beta: float = 0.4, solver: str = 'gurobi'):
        self.alpha = alpha
        self.penalty = penalty
        self.beta = beta
        self.solver = solver

    def retrieve(self, query_embedding: np.ndarray, candidates: List[Dict[str, Any]], k: int) -> List[RetrievalResult]:
        candidate_embeddings = np.array([c['embedding'] for c in candidates])
        query_sims = np.array([c['score'] for c in candidates])

        n = len(candidates)
        pairwise_sim = compute_pairwise_similarities(candidate_embeddings)

        # Apply the beta threshold
        thresholded_pairwise_sim = np.where(pairwise_sim >= self.beta, pairwise_sim, 0)

        if self.solver == 'gurobi':
            selected_indices = self._solve_gurobi(query_sims, thresholded_pairwise_sim, k, n)
        elif self.solver == 'orbit':
            selected_indices = self._solve_orbit(query_sims, thresholded_pairwise_sim, k, n)
        else:
            raise ValueError(f"Unknown solver: {self.solver}. Use 'gurobi' or 'orbit'.")

        return [RetrievalResult(Chunk(id=candidates[i]['id'], text=candidates[i]['text'],
                                      source=candidates[i]['metadata'].get('source', candidates[i]['metadata'].get('article_title', 'unknown')),
                                      metadata=candidates[i]['metadata']),
                                candidates[i]['score'], rank+1)
                for rank, i in enumerate(selected_indices)]

    def _solve_gurobi(self, query_sims: np.ndarray, pairwise_sim: np.ndarray, k: int, n: int) -> List[int]:
        """Solve QUBO using Gurobi."""
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError:
            raise ImportError("Gurobi not found. Please install it to use solver='gurobi'.")

        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            with gp.Model("QUBO_Retrieval", env=env) as model:
                x = model.addMVar(shape=n, vtype=GRB.BINARY, name="x")

                relevance_term = -query_sims @ x
                diversity_term = self.alpha * (x @ (pairwise_sim - np.identity(n)) @ x)

                model.setObjective(relevance_term + diversity_term, GRB.MINIMIZE)
                model.addConstr(x.sum() == k, "cardinality")
                model.optimize()

                if model.Status == GRB.OPTIMAL:
                    return [i for i, v in enumerate(x.X) if v > 0.5]
                else:
                    # Fallback to naive
                    return list(range(k))

    def _solve_orbit(self, query_sims: np.ndarray, pairwise_sim: np.ndarray, k: int, n: int) -> List[int]:
        """Solve QUBO using ORBIT p-bit simulator."""
        try:
            import orbit
        except ImportError:
            raise ImportError("ORBIT not found. Please install it to use solver='orbit'.")

        # Convert QUBO to Ising format
        # Note: pairwise_sim already has diagonal zeroed via (pairwise_sim - I) in objective
        # So we use pairwise_sim - I for the diversity matrix S
        S = pairwise_sim - np.identity(n)
        J, h = qubo_to_ising(query_sims, S, self.alpha, self.penalty, k)

        # Scale coefficients to prevent overflow in ORBIT's sigmoid function
        # ORBIT uses sigmoid(bias) which overflows if |bias| is too large
        # Scale by max absolute value to keep coefficients reasonable
        max_J = np.max(np.abs(J)) if J.size > 0 else 1.0
        max_h = np.max(np.abs(h)) if h.size > 0 else 1.0
        scale = max(max_J, max_h)

        if scale > 10.0:  # Only scale if needed
            J_scaled = J / scale
            h_scaled = h / scale
        else:
            J_scaled = J
            h_scaled = h

        # Solve with ORBIT
        # Beta range: start low for exploration, end high for exploitation
        result = orbit.optimize_ising(
            J_scaled, h_scaled,
            n_replicas=4,
            full_sweeps=15000,
            beta_initial=0.1,
            beta_end=5.0,
            beta_step_interval=1
        )

        # Convert Ising solution back to binary variables
        x = ising_to_qubo_solution(result.min_state)

        # Extract selected indices (where x_i = 1)
        selected_indices = [i for i, val in enumerate(x) if val == 1]

        # If not exactly k selected (rare), fall back to top k by relevance
        if len(selected_indices) != k:
            sorted_indices = np.argsort(-query_sims)
            return sorted_indices[:k].tolist()

        return selected_indices

class Retriever:
    """Handles the retrieval process."""
    def __init__(self, embedder: EmbeddingGenerator, vector_store: VectorStore):
        self.embedder = embedder
        self.vector_store = vector_store
        self.strategies = {
            'naive': NaiveRetrieval(),
            'mmr': MMRRetrieval(),
            'qubo': QUBORetrieval()
        }

    def retrieve(self, query: str, k: int = 5, strategy: str = 'naive', candidates: Optional[List[Dict[str, Any]]] = None, **kwargs) -> List[RetrievalResult]:
        """Retrieves chunks based on the selected strategy."""
        if strategy not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy}. Available: {list(self.strategies.keys())}")

        query_embedding = self.embedder.embed([query])[0]
        
        if candidates is None:
            # Fetch more candidates for diversity-aware strategies
            candidate_multiplier = 3 if strategy in ['mmr', 'qubo'] else 1
            candidates = self.vector_store.search(query_embedding, k=k * candidate_multiplier)
        
        retrieval_strategy = self.strategies[strategy]
        if isinstance(retrieval_strategy, (MMRRetrieval, QUBORetrieval)):
            for key, value in kwargs.items():
                if hasattr(retrieval_strategy, key):
                    setattr(retrieval_strategy, key, value)

        return retrieval_strategy.retrieve(query_embedding, candidates, k)
