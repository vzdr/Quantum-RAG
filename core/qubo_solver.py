"""
QUBO Solver Module - Wrapper for backwards compatibility with submission notebooks.
"""
import numpy as np
from typing import List, Tuple, Dict, Any
import gurobipy as gp
from gurobipy import GRB
from .utils import compute_pairwise_similarities


def solve_diverse_retrieval_qubo(
    query_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    k: int,
    alpha: float = 0.20,
    penalty: float = 1000.0,
    solver: str = 'gurobi',
    solver_options: Dict[str, Any] = None
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Solve QUBO for diverse retrieval.

    Args:
        query_embedding: Query vector
        candidate_embeddings: Array of candidate embedding vectors (n x d)
        k: Number of items to retrieve
        alpha: Diversity weight
        penalty: Cardinality constraint penalty
        solver: Solver to use ('gurobi')
        solver_options: Additional solver options

    Returns:
        selected_indices: List of selected indices
        metadata: Solver metadata
    """
    solver_options = solver_options or {}

    # Compute similarities
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    cand_norms = candidate_embeddings / np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
    query_sims = np.dot(cand_norms, query_norm)

    # Compute pairwise similarities
    pairwise_sim = compute_pairwise_similarities(candidate_embeddings)

    n = len(candidate_embeddings)

    # Solve with Gurobi
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', solver_options.get('OutputFlag', 0))
        env.start()
        with gp.Model("QUBO_Retrieval", env=env) as model:
            x = model.addMVar(shape=n, vtype=GRB.BINARY, name="x")

            relevance_term = -query_sims @ x
            diversity_term = alpha * (x @ (pairwise_sim - np.identity(n)) @ x)

            model.setObjective(relevance_term + diversity_term, GRB.MINIMIZE)
            model.addConstr(x.sum() == k, "cardinality")
            model.optimize()

            if model.Status == GRB.OPTIMAL:
                selected_indices = [i for i, v in enumerate(x.X) if v > 0.5]
                energy = model.objVal
                solve_time = model.Runtime
            else:
                # Fallback to naive
                selected_indices = list(range(min(k, n)))
                energy = 0.0
                solve_time = 0.0

    # Compute solution quality
    avg_relevance = float(np.mean(query_sims[selected_indices])) if selected_indices else 0.0

    metadata = {
        'energy': energy,
        'execution_time': solve_time,
        'solution_quality': {
            'avg_relevance': avg_relevance
        }
    }

    return selected_indices, metadata
