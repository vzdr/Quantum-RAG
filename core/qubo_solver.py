"""
QUBO Solver Module

Implements QUBO-based diverse retrieval optimization using ORBIT simulator.

Components:
- QUBOProblem: Constructs QUBO matrix from query and candidate embeddings
- IsingConverter: Converts between QUBO and Ising formulations
- ORBITSolver: Wrapper for ORBIT p-bit simulator
- Helper functions for constraint enforcement
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np


class QUBOProblem:
    """
    Constructs and manages QUBO problem for diverse retrieval.

    Formulation:
        minimize f(x) = -α * Σᵢ sim(query, chunkᵢ) * xᵢ
                        + (1-α) * Σᵢ Σⱼ>ᵢ sim(chunkᵢ, chunkⱼ) * xᵢ * xⱼ
        subject to: Σxᵢ = k

    where:
        - xᵢ ∈ {0,1}: binary decision variable
        - α ∈ [0.5, 0.7]: relevance vs diversity trade-off
        - Linear term: rewards query-chunk similarity (relevance)
        - Quadratic term: penalizes inter-chunk similarity (diversity)
    """

    def __init__(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        alpha: float,
        k: int,
        penalty_multiplier: float = 2.0
    ):
        """
        Initialize QUBO problem.

        Args:
            query_embedding: Query vector (embedding_dim,)
            candidate_embeddings: Candidate vectors (n_candidates, embedding_dim)
            alpha: Relevance weight (0.5-0.7), (1-alpha) = diversity weight
            k: Number of items to select
            penalty_multiplier: Multiplier for cardinality penalty (default: 2.0)
                              Higher = stricter constraint enforcement
        """
        # Validate alpha parameter
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Alpha must be in [0, 1], got {alpha}")
        if not 0.3 <= alpha <= 0.7:
            import warnings
            warnings.warn(
                f"Alpha={alpha} is outside typical range [0.3, 0.7]. "
                f"Typical usage: 0.5-0.6 for balanced, <0.5 for more diversity, >0.6 for more relevance."
            )

        self.query_embedding = query_embedding
        self.candidate_embeddings = candidate_embeddings
        self.alpha = alpha
        self.k = k
        self.n = len(candidate_embeddings)
        self.penalty_multiplier = penalty_multiplier

        # Cache similarity matrices for diagnostics
        self.query_sims = None
        self.pairwise_sims = None

    def build_qubo_matrix(self) -> np.ndarray:
        """
        Construct QUBO matrix Q (n × n) from problem formulation.

        Returns:
            Q: Symmetric QUBO matrix with cardinality constraint encoded
        """
        Q = np.zeros((self.n, self.n))

        # Compute similarity matrices and cache for diagnostics
        self.query_sims = self._compute_query_similarities()
        self.pairwise_sims = self._compute_pairwise_similarities()

        # Estimate penalty parameter for cardinality constraint
        penalty_lambda = self._estimate_penalty_parameter()

        # Build diagonal (linear terms + cardinality penalty)
        for i in range(self.n):
            # Original QUBO: -α * query_sim[i]
            Q[i, i] = -self.alpha * self.query_sims[i]

            # Cardinality penalty: λ * (1 - 2k)
            Q[i, i] += penalty_lambda * (1 - 2 * self.k)

        # Build off-diagonal (quadratic terms + cardinality penalty)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                # Original QUBO: (1-α) * pairwise_sim[i,j]
                Q[i, j] = (1 - self.alpha) * self.pairwise_sims[i, j]

                # Cardinality penalty: 2λ
                Q[i, j] += 2 * penalty_lambda

                # Ensure symmetry
                Q[j, i] = Q[i, j]

        return Q

    def _compute_query_similarities(self) -> np.ndarray:
        """
        Compute cosine similarity between query and all candidates.

        Returns:
            Similarity scores (n_candidates,)
        """
        # Normalize vectors
        query_norm = self.query_embedding / np.linalg.norm(self.query_embedding)
        cand_norms = self.candidate_embeddings / np.linalg.norm(
            self.candidate_embeddings, axis=1, keepdims=True
        )

        # Cosine similarity
        return np.dot(cand_norms, query_norm)

    def _compute_pairwise_similarities(self) -> np.ndarray:
        """
        Compute pairwise cosine similarity matrix for candidates.

        Returns:
            Similarity matrix (n_candidates, n_candidates)
        """
        # Normalize vectors
        norms = self.candidate_embeddings / np.linalg.norm(
            self.candidate_embeddings, axis=1, keepdims=True
        )

        # Pairwise cosine similarity
        return np.dot(norms, norms.T)

    def _estimate_penalty_parameter(self) -> float:
        """
        Estimate penalty parameter λ for cardinality constraint.

        Rule: λ should dominate other terms to enforce constraint.
        Strategy: Use penalty_multiplier * max_similarity_value

        The penalty_multiplier controls how strictly the cardinality constraint
        is enforced:
        - Higher values (2.5-3.0): Stricter enforcement, may reduce diversity
        - Lower values (1.5-2.0): Softer enforcement, may violate constraint

        Returns:
            Penalty parameter λ
        """
        max_query_sim = np.max(np.abs(self.query_sims))
        max_pair_sim = np.max(np.abs(self.pairwise_sims))
        return self.penalty_multiplier * max(max_query_sim, max_pair_sim)

    def compute_solution_quality(self, solution: np.ndarray, Q: np.ndarray) -> Dict[str, float]:
        """
        Compute quality metrics for a QUBO solution.

        Diagnostics help understand solver performance and parameter tuning:
        - QUBO energy: Lower is better (minimization problem)
        - Constraint violation: Should be 0 for exact k selections
        - Intra-list similarity: Lower = more diverse
        - Average relevance: Higher = more relevant to query

        Args:
            solution: Binary solution vector {0, 1}^n
            Q: QUBO matrix used to generate solution

        Returns:
            Dictionary with quality metrics
        """
        # QUBO energy (objective value)
        energy = float(solution @ Q @ solution)

        # Cardinality constraint check
        k_actual = int(np.sum(solution))
        constraint_violation = abs(k_actual - self.k)

        # Get selected indices
        selected_indices = np.where(solution == 1)[0]

        # Compute diversity of selected chunks (intra-list similarity)
        if len(selected_indices) > 1:
            selected_pairwise = self.pairwise_sims[selected_indices][:, selected_indices]
            # Average off-diagonal elements
            n = len(selected_indices)
            intra_similarity = (np.sum(selected_pairwise) - n) / (n * (n - 1))
        else:
            intra_similarity = 0.0

        # Average relevance of selected chunks
        if len(selected_indices) > 0:
            avg_relevance = float(np.mean(self.query_sims[selected_indices]))
        else:
            avg_relevance = 0.0

        return {
            'qubo_energy': energy,
            'constraint_violation': constraint_violation,
            'k_actual': k_actual,
            'k_target': self.k,
            'intra_list_similarity': float(intra_similarity),
            'avg_relevance': avg_relevance,
            'min_relevance': float(np.min(self.query_sims[selected_indices])) if len(selected_indices) > 0 else 0.0,
            'max_relevance': float(np.max(self.query_sims[selected_indices])) if len(selected_indices) > 0 else 0.0,
        }


class IsingConverter:
    """
    Converts between QUBO and Ising formulations for ORBIT.

    ORBIT expects Ising format:
        H_Ising = Σᵢⱼ J[i,j] * sᵢ * sⱼ + Σᵢ h[i] * sᵢ

    where sᵢ ∈ {-1, 1}

    Conversion from QUBO (xᵢ ∈ {0, 1}) to Ising:
        xᵢ = (sᵢ + 1) / 2
    """

    @staticmethod
    def qubo_to_ising(Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Convert QUBO matrix to Ising parameters.

        Mapping: xᵢ ∈ {0,1} → sᵢ ∈ {-1,1} via xᵢ = (sᵢ + 1) / 2

        QUBO: Σᵢⱼ Q[i,j] * xᵢ * xⱼ
        Ising: Σᵢⱼ J[i,j] * sᵢ * sⱼ + Σᵢ h[i] * sᵢ + offset

        Conversion formulas:
            J[i,j] = Q[i,j] / 4  (for i ≠ j)
            h[i] = (Σⱼ Q[i,j]) / 2 - Q[i,i] / 2
            offset = Σᵢ Q[i,i] / 4 + Σᵢⱼ (i<j) Q[i,j] / 4

        Args:
            Q: QUBO matrix (n × n, symmetric)

        Returns:
            J: Coupling matrix (n × n)
            h: External field vector (n,)
            offset: Constant energy offset
        """
        n = Q.shape[0]

        # Coupling matrix J
        J = Q / 4.0
        np.fill_diagonal(J, 0)  # No self-coupling in Ising

        # External field h
        h = np.zeros(n)
        for i in range(n):
            h[i] = np.sum(Q[i, :]) / 2.0 - Q[i, i] / 2.0

        # Constant offset
        offset = np.sum(np.diag(Q)) / 4.0
        for i in range(n):
            for j in range(i + 1, n):
                offset += Q[i, j] / 4.0

        return J, h, offset

    @staticmethod
    def ising_to_binary(ising_state: np.ndarray) -> np.ndarray:
        """
        Convert Ising solution {-1, 1} to binary {0, 1}.

        Mapping: xᵢ = (sᵢ + 1) / 2

        Args:
            ising_state: Ising solution vector {-1, 1}^n

        Returns:
            Binary solution vector {0, 1}^n
        """
        return ((ising_state + 1) / 2).astype(int)


class ORBITSolver:
    """
    Wrapper for ORBIT p-bit simulator.

    ORBIT is a probabilistic computing simulator that solves Ising
    problems using parallel tempering Monte Carlo.
    """

    def __init__(
        self,
        n_replicas: int = 4,
        full_sweeps: int = 10000,
        beta_initial: float = 0.35,
        beta_end: float = 3.5,
        beta_step_interval: int = 1,
        max_processes: Optional[int] = None
    ):
        """
        Initialize ORBIT solver parameters.

        Args:
            n_replicas: Number of parallel replicas (default: 4)
            full_sweeps: Number of annealing sweeps (default: 10000)
                        Higher = better quality, slower
            beta_initial: Initial inverse temperature (default: 0.35)
                         Low = high exploration
            beta_end: Final inverse temperature (default: 3.5)
                     High = exploitation/convergence
            beta_step_interval: Sweeps per temperature step (default: 1)
            max_processes: Max CPU cores to use (default: auto)
        """
        self.n_replicas = n_replicas
        self.full_sweeps = full_sweeps
        self.beta_initial = beta_initial
        self.beta_end = beta_end
        self.beta_step_interval = beta_step_interval
        self.max_processes = max_processes

    def solve(self, J: np.ndarray, h: np.ndarray) -> Dict[str, Any]:
        """
        Solve Ising problem using ORBIT.

        Args:
            J: Coupling matrix (n × n)
            h: External field vector (n,)

        Returns:
            Dictionary with:
                - 'binary_solution': Binary solution vector {0,1}^n
                - 'ising_solution': Ising solution vector {-1,1}^n
                - 'energy': Final energy
                - 'execution_time': Solver runtime in seconds
        """
        try:
            import orbit
        except ImportError:
            raise ImportError(
                "ORBIT not found. Please install with: "
                "uv pip install path/to/orbit-0.2.0-py3-none-any.whl"
            )

        # Prepare solver parameters
        solver_params = {
            'J': J,
            'h': h,
            'n_replicas': self.n_replicas,
            'full_sweeps': self.full_sweeps,
            'beta_initial': self.beta_initial,
            'beta_end': self.beta_end,
            'beta_step_interval': self.beta_step_interval,
        }

        if self.max_processes is not None:
            solver_params['max_processes'] = self.max_processes

        # Run ORBIT optimization
        result = orbit.optimize_ising(**solver_params)

        # Convert Ising solution to binary
        binary_solution = IsingConverter.ising_to_binary(result.min_state)

        return {
            'binary_solution': binary_solution,
            'ising_solution': result.min_state,
            'energy': result.min_cost,
            'execution_time': result.execution_time if hasattr(result, 'execution_time') else None
        }


def solve_diverse_retrieval_qubo(
    query_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    k: int,
    alpha: float = 0.6,
    penalty_multiplier: float = 2.0,
    solver_params: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    End-to-end QUBO solver for diverse retrieval.

    Pipeline:
        1. Build QUBO matrix from embeddings
        2. Convert QUBO to Ising format
        3. Solve with ORBIT simulator
        4. Extract selected indices
        5. Enforce cardinality constraint if violated

    Args:
        query_embedding: Query vector (embedding_dim,)
        candidate_embeddings: Candidate vectors (n_candidates, embedding_dim)
        k: Number of items to select
        alpha: Relevance weight (0.5-0.7), (1-alpha) = diversity weight
        penalty_multiplier: Cardinality constraint penalty multiplier (default: 2.0)
        solver_params: Optional ORBIT parameters (n_replicas, full_sweeps, etc.)

    Returns:
        selected_indices: Indices of selected items (k,)
        metadata: Dictionary with timing, energy, constraint satisfaction info
    """
    # Build QUBO
    problem = QUBOProblem(query_embedding, candidate_embeddings, alpha, k, penalty_multiplier)
    Q = problem.build_qubo_matrix()

    # Convert to Ising
    J, h, offset = IsingConverter.qubo_to_ising(Q)

    # Solve with ORBIT
    solver_params = solver_params or {}
    solver = ORBITSolver(**solver_params)
    result = solver.solve(J, h)

    # Extract selected indices
    binary_sol = result['binary_solution']
    selected_indices = np.where(binary_sol == 1)[0]

    # Check cardinality constraint
    constraint_satisfied = (len(selected_indices) == k)

    # Enforce constraint if violated
    adjusted_solution = binary_sol.copy()
    if not constraint_satisfied:
        selected_indices = _adjust_cardinality(
            binary_sol, k, query_embedding, candidate_embeddings
        )
        # Update adjusted solution
        adjusted_solution = np.zeros_like(binary_sol)
        adjusted_solution[selected_indices] = 1

    # Compute solution quality diagnostics
    quality_metrics = problem.compute_solution_quality(adjusted_solution, Q)

    # Compute metadata
    metadata = {
        'qubo_energy': quality_metrics['qubo_energy'],
        'ising_energy': result['energy'],
        'execution_time': result['execution_time'],
        'constraint_satisfied': constraint_satisfied,
        'alpha': alpha,
        'penalty_multiplier': penalty_multiplier,
        'k': k,
        'n_candidates': len(candidate_embeddings),
        # Add quality diagnostics
        'solution_quality': quality_metrics
    }

    return selected_indices, metadata


def _adjust_cardinality(
    binary_sol: np.ndarray,
    target_k: int,
    query_emb: np.ndarray,
    candidate_embs: np.ndarray
) -> np.ndarray:
    """
    Adjust solution to satisfy cardinality constraint via greedy selection.

    Strategy:
        - If too many selected: Remove lowest query similarity items
        - If too few selected: Add highest query similarity items from unselected

    Args:
        binary_sol: Current binary solution
        target_k: Desired number of selections
        query_emb: Query embedding
        candidate_embs: Candidate embeddings

    Returns:
        Adjusted indices (exactly target_k items)
    """
    selected = np.where(binary_sol == 1)[0]

    if len(selected) > target_k:
        # Remove lowest relevance items
        query_sims = _compute_cosine_similarities(
            query_emb, candidate_embs[selected]
        )
        keep_indices = np.argsort(query_sims)[-target_k:]
        return selected[keep_indices]

    elif len(selected) < target_k:
        # Add highest relevance items from unselected
        unselected = np.where(binary_sol == 0)[0]
        query_sims = _compute_cosine_similarities(
            query_emb, candidate_embs[unselected]
        )
        n_to_add = target_k - len(selected)
        add_indices = unselected[np.argsort(query_sims)[-n_to_add:]]
        return np.concatenate([selected, add_indices])

    return selected


def _compute_cosine_similarities(
    query_emb: np.ndarray,
    candidate_embs: np.ndarray
) -> np.ndarray:
    """
    Compute cosine similarity between query and candidates.

    Args:
        query_emb: Query vector (d,)
        candidate_embs: Candidate vectors (n, d)

    Returns:
        Similarities (n,)
    """
    query_norm = query_emb / np.linalg.norm(query_emb)
    cand_norms = candidate_embs / np.linalg.norm(
        candidate_embs, axis=1, keepdims=True
    )
    return np.dot(cand_norms, query_norm)


def _evaluate_qubo_energy(Q: np.ndarray, x: np.ndarray) -> float:
    """
    Evaluate QUBO objective: x^T Q x

    Args:
        Q: QUBO matrix
        x: Binary solution vector

    Returns:
        QUBO energy (objective value)
    """
    return float(x @ Q @ x)
