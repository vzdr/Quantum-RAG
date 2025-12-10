"""
QUBO Solver Module

Solves diverse retrieval as QUBO using the direct formula:
    Energy = alpha * s^T Q s - h^T s + P * (s^T 1 - k)^2

Where:
    Q: chunk-to-chunk similarity matrix (n x n)
    h: chunk-to-query similarity vector (n,)
    s: binary selection vector {0,1}^n
    alpha: diversity weight
    P: cardinality penalty
    k: number of selections
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np


def compute_cosine_similarities(
    query_emb: np.ndarray,
    candidate_embs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute similarity vectors and matrices.

    Args:
        query_emb: Query embedding (d,)
        candidate_embs: Candidate embeddings (n, d)

    Returns:
        h: Query-to-chunk similarities (n,)
        Q: Chunk-to-chunk similarities (n, n)
    """
    # Normalize
    query_norm = query_emb / np.linalg.norm(query_emb)
    cand_norms = candidate_embs / np.linalg.norm(candidate_embs, axis=1, keepdims=True)

    # h vector: query similarities
    h = np.dot(cand_norms, query_norm)

    # Q matrix: pairwise similarities
    Q = np.dot(cand_norms, cand_norms.T)

    return h, Q


def evaluate_energy(s: np.ndarray, Q: np.ndarray, h: np.ndarray, alpha: float, P: float, k: int) -> float:
    """
    Evaluate the energy function: alpha * s^T Q s - h^T s + P * (s^T 1 - k)^2

    Args:
        s: Binary selection vector
        Q: Pairwise similarity matrix
        h: Query similarity vector
        alpha: Diversity weight
        P: Penalty weight
        k: Target selections

    Returns:
        Energy value
    """
    diversity_term = alpha * (s @ Q @ s)
    relevance_term = -(h @ s)
    cardinality_term = P * (np.sum(s) - k) ** 2

    return diversity_term + relevance_term + cardinality_term


def solve_with_bruteforce(Q: np.ndarray, h: np.ndarray, k: int, alpha: float, P: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Solve by brute-force enumeration.

    Args:
        Q: Pairwise similarity matrix
        h: Query similarity vector
        k: Number of items to select
        alpha: Diversity weight
        P: Penalty weight

    Returns:
        solution: Binary solution vector
        metadata: Solver information
    """
    import time
    from itertools import combinations

    n = len(h)
    if n > 25:
        raise ValueError(f"Brute-force infeasible for n={n} (max 25)")

    start = time.time()
    best_energy = float('inf')
    best_solution = None

    # Try all C(n,k) combinations
    for selected in combinations(range(n), k):
        s = np.zeros(n, dtype=int)
        s[list(selected)] = 1

        energy = evaluate_energy(s, Q, h, alpha, P, k)

        if energy < best_energy:
            best_energy = energy
            best_solution = s

    metadata = {
        'solver': 'bruteforce',
        'energy': best_energy,
        'execution_time': time.time() - start
    }

    return best_solution, metadata


def solve_with_orbit(Q: np.ndarray, h: np.ndarray, k: int, alpha: float, P: float, **orbit_params) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Solve using ORBIT by converting to QUBO matrix form.

    The formula: alpha * s^T Q s - h^T s + P * (s^T 1 - k)^2
    needs to be converted to standard QUBO form: s^T M s

    Expanding (s^T 1 - k)^2:
        (s^T 1 - k)^2 = s^T 1 1^T s - 2k * s^T 1 + k^2
                      = sum_ij(s_i * s_j) - 2k * sum_i(s_i) + k^2

    For binary s where s_i^2 = s_i:
        sum_ij(s_i * s_j) = sum_i(s_i) + sum_{i!=j}(s_i * s_j)

    So the full formula in matrix form s^T M s is:
        M_ij = alpha * Q_ij + P  (for i != j)
        M_ii = -h_i + P * (1 - 2k)  (diagonal)

    Args:
        Q: Pairwise similarity matrix
        h: Query similarity vector
        k: Number of selections
        alpha: Diversity weight
        P: Penalty weight
        **orbit_params: ORBIT solver parameters

    Returns:
        solution: Binary solution vector
        metadata: Solver information
    """
    try:
        import orbit
    except ImportError:
        raise ImportError("ORBIT not installed")

    n = len(h)

    # Build QUBO matrix M
    M = np.zeros((n, n))

    # Off-diagonal: alpha * Q[i,j] + P
    M = alpha * Q + P * np.ones((n, n))

    # Diagonal: -h_i + P * (1 - 2k)
    np.fill_diagonal(M, -h + P * (1 - 2 * k))

    # Convert QUBO to Ising for ORBIT
    # QUBO: E = s^T M s where s ∈ {0,1}
    # Ising: E = sigma^T J sigma + h_ising^T sigma where sigma ∈ {-1,1}
    # Mapping: s = (sigma + 1) / 2

    J = M / 4.0
    np.fill_diagonal(J, 0)  # No self-coupling

    h_ising = np.sum(M, axis=1) / 2.0 - np.diag(M) / 2.0

    offset = np.sum(np.diag(M)) / 4.0 + np.sum(np.triu(M, k=1)) / 4.0

    # Default ORBIT parameters
    params = {
        'n_replicas': 4,
        'full_sweeps': 10000,
        'beta_initial': 0.35,
        'beta_end': 3.5,
        'beta_step_interval': 1
    }
    params.update(orbit_params)

    # Solve
    result = orbit.optimize_ising(
        J=J,
        h=h_ising,
        n_replicas=params['n_replicas'],
        full_sweeps=params['full_sweeps'],
        beta_initial=params['beta_initial'],
        beta_end=params['beta_end'],
        beta_step_interval=params['beta_step_interval']
    )

    # Convert back to binary
    sigma = result.min_state
    s = ((sigma + 1) / 2).astype(int)

    # Enforce exactly k selections
    s = enforce_cardinality(s, k, h)

    energy = evaluate_energy(s, Q, h, alpha, P, k)

    metadata = {
        'solver': 'orbit',
        'energy': energy,
        'ising_energy': result.min_cost,
        'execution_time': getattr(result, 'execution_time', None),
        **params
    }

    return s, metadata


def solve_with_gurobi(Q: np.ndarray, h: np.ndarray, k: int, alpha: float, P: float, time_limit: int = 10) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Solve using Gurobi with hard cardinality constraint.

    Args:
        Q: Pairwise similarity matrix
        h: Query similarity vector
        k: Number of selections
        alpha: Diversity weight
        P: Penalty weight (not used, constraint is hard)
        time_limit: Solver time limit

    Returns:
        solution: Binary solution vector
        metadata: Solver information
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        raise ImportError("Gurobi not installed")

    import time
    n = len(h)
    start = time.time()

    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()

        with gp.Model(env=env) as model:
            # Variables
            s = model.addMVar(shape=n, vtype=GRB.BINARY, name="s")

            # Objective: alpha * s^T Q s - h^T s
            # (no penalty term needed, we use hard constraint)
            model.setObjective(alpha * s @ Q @ s - h @ s, GRB.MINIMIZE)

            # Hard cardinality constraint
            model.addConstr(s.sum() == k, "cardinality")

            # Solve
            model.setParam('TimeLimit', time_limit)
            model.optimize()

            if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
                solution = np.array(s.X).astype(int)
                energy = evaluate_energy(solution, Q, h, alpha, P, k)
            else:
                solution = np.zeros(n, dtype=int)
                energy = float('inf')

            metadata = {
                'solver': 'gurobi',
                'energy': energy,
                'execution_time': time.time() - start,
                'optimal': model.Status == GRB.OPTIMAL
            }

            return solution, metadata


def enforce_cardinality(s: np.ndarray, k: int, h: np.ndarray) -> np.ndarray:
    """
    Adjust solution to have exactly k selections based on relevance.

    Args:
        s: Binary solution vector
        k: Target number of selections
        h: Query similarity vector (for greedy adjustment)

    Returns:
        Adjusted solution with exactly k selections
    """
    selected = np.where(s == 1)[0]
    current_k = len(selected)

    if current_k == k:
        return s

    s = s.copy()

    if current_k > k:
        # Remove items with lowest relevance
        remove_count = current_k - k
        to_remove = selected[np.argsort(h[selected])[:remove_count]]
        s[to_remove] = 0

    elif current_k < k:
        # Add items with highest relevance
        unselected = np.where(s == 0)[0]
        add_count = k - current_k
        to_add = unselected[np.argsort(h[unselected])[-add_count:]]
        s[to_add] = 1

    return s


# Solver presets for ORBIT
ORBIT_PRESETS = {
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
        'beta_initial': 0.2,
        'beta_end': 4.0,
        'beta_step_interval': 2
    }
}


def solve_diverse_retrieval_qubo(
    query_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    k: int,
    alpha: float = 0.05,
    penalty: float = 1000.0,
    solver: str = 'orbit',
    solver_options: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Solve diverse retrieval as QUBO.

    Energy = alpha * s^T Q s - h^T s + P * (s^T 1 - k)^2

    Args:
        query_embedding: Query vector (d,)
        candidate_embeddings: Candidate vectors (n, d)
        k: Number of items to select
        alpha: Diversity weight
        penalty: Cardinality penalty
        solver: 'orbit', 'bruteforce', or 'gurobi'
        solver_options: Solver-specific options

    Returns:
        selected_indices: Indices of selected items
        metadata: Solver metadata
    """
    solver_options = solver_options or {}

    # Compute similarities
    h, Q = compute_cosine_similarities(query_embedding, candidate_embeddings)

    # Solve with selected solver
    if solver == 'orbit':
        # Get preset or custom params
        preset = solver_options.pop('preset', 'balanced')
        params = ORBIT_PRESETS.get(preset, ORBIT_PRESETS['balanced']).copy()
        params.update(solver_options)

        solution, metadata = solve_with_orbit(Q, h, k, alpha, penalty, **params)

    elif solver == 'bruteforce':
        solution, metadata = solve_with_bruteforce(Q, h, k, alpha, penalty)

    elif solver == 'gurobi':
        time_limit = solver_options.get('time_limit', 10)
        solution, metadata = solve_with_gurobi(Q, h, k, alpha, penalty, time_limit)

    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Get selected indices
    selected_indices = np.where(solution == 1)[0]

    # Compute metrics
    n_selected = len(selected_indices)
    if n_selected > 0:
        avg_relevance = float(np.mean(h[selected_indices]))

        if n_selected > 1:
            selected_sims = Q[selected_indices][:, selected_indices]
            intra_sim = (np.sum(selected_sims) - n_selected) / (n_selected * (n_selected - 1))
        else:
            intra_sim = 0.0
    else:
        avg_relevance = intra_sim = 0.0

    metadata.update({
        'alpha': alpha,
        'penalty': penalty,
        'n_candidates': len(candidate_embeddings),
        'solution_quality': {
            'n_selected': n_selected,
            'constraint_violation': abs(n_selected - k),
            'avg_relevance': avg_relevance,
            'intra_list_similarity': float(intra_sim)
        }
    })

    return selected_indices, metadata
