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
    """QUBO-based retrieval using Gurobi or other solvers."""
    def __init__(self, alpha: float = 0.04, penalty: float = 10.0, beta: float = 0.4, solver: str = 'gurobi'):
        self.alpha = alpha
        self.penalty = penalty
        self.beta = beta
        self.solver = solver

    def retrieve(self, query_embedding: np.ndarray, candidates: List[Dict[str, Any]], k: int) -> List[RetrievalResult]:
        candidate_embeddings = np.array([c['embedding'] for c in candidates])
        query_sims = np.array([c['score'] for c in candidates])
        
        # This is where the QUBO formulation and solver call would go.
        # For now, we'll mock the Gurobi solver logic.
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError:
            raise ImportError("Gurobi not found. Please install it to use the QUBO strategy.")

        n = len(candidates)
        pairwise_sim = compute_pairwise_similarities(candidate_embeddings)
        
        # Apply the beta threshold
        thresholded_pairwise_sim = np.where(pairwise_sim >= self.beta, pairwise_sim, 0)
        
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            with gp.Model("QUBO_Retrieval", env=env) as model:
                x = model.addMVar(shape=n, vtype=GRB.BINARY, name="x")
                
                relevance_term = -query_sims @ x
                diversity_term = self.alpha * (x @ (thresholded_pairwise_sim - np.identity(n)) @ x)
                
                model.setObjective(relevance_term + diversity_term, GRB.MINIMIZE)
                model.addConstr(x.sum() == k, "cardinality")
                model.optimize()

                if model.Status == GRB.OPTIMAL:
                    selected_indices = [i for i, v in enumerate(x.X) if v > 0.5]
                else: # Fallback to naive if solver fails
                    selected_indices = list(range(k))

        return [RetrievalResult(Chunk(id=candidates[i]['id'], text=candidates[i]['text'], source=candidates[i]['metadata'].get('source', candidates[i]['metadata'].get('article_title', 'unknown')), metadata=candidates[i]['metadata']), candidates[i]['score'], rank+1) for rank, i in enumerate(selected_indices)]

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
