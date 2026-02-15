"""
Analysis utilities for evaluating retrieval quality.

Provides functions for computing similarity matrices and evaluating
retrieval results with metrics like gold recall and redundancy.
"""

import numpy as np
from typing import List, Dict


def compute_pairwise_similarities(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarities between all embeddings.

    Args:
        embeddings: Document embeddings (shape: [n_docs, embedding_dim])

    Returns:
        Similarity matrix (shape: [n_docs, n_docs])
        Element [i, j] = cosine similarity between doc i and doc j
    """
    # Normalize embeddings to unit vectors (for cosine similarity)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / (norms + 1e-8)  # Add epsilon to avoid division by zero

    # Compute dot product = cosine similarity for normalized vectors
    similarity_matrix = normalized_embeddings @ normalized_embeddings.T

    return similarity_matrix


def evaluate_retrieval_quality(
    selected_indices: List[int],
    gold_indices: List[int],
    similarity_matrix: np.ndarray
) -> Dict:
    """
    Evaluate quality metrics for a retrieval result.

    Computes standard RAG evaluation metrics:
    - Gold recall: Percentage of gold documents retrieved
    - Average redundancy: Mean pairwise similarity among retrieved docs
    - Gold percentage: Percentage of retrieved docs that are gold

    Args:
        selected_indices: Indices of selected documents
        gold_indices: Indices of gold (ground truth) documents
        similarity_matrix: Pairwise document similarities

    Returns:
        Dictionary with metrics:
        {
            'gold_recall': float,      # % of gold docs retrieved (0-100)
            'avg_redundancy': float,   # Mean similarity in retrieved set (0-1)
            'gold_percentage': float   # % of retrieved docs that are gold (0-100)
        }
    """
    gold_set = set(gold_indices)
    selected_set = set(selected_indices)

    # Gold recall: what percentage of gold docs were retrieved?
    retrieved_gold = gold_set & selected_set
    gold_recall = 100.0 * len(retrieved_gold) / len(gold_set) if gold_set else 0.0

    # Gold percentage: what percentage of retrieved docs are gold?
    gold_percentage = 100.0 * len(retrieved_gold) / len(selected_set) if selected_set else 0.0

    # Average redundancy: mean pairwise similarity among retrieved docs
    if len(selected_indices) > 1:
        redundancies = []
        for i, idx_i in enumerate(selected_indices):
            for idx_j in selected_indices[i+1:]:
                redundancies.append(similarity_matrix[idx_i, idx_j])
        avg_redundancy = np.mean(redundancies)
    else:
        avg_redundancy = 0.0

    return {
        'gold_recall': gold_recall,
        'avg_redundancy': avg_redundancy,
        'gold_percentage': gold_percentage
    }


def compute_qubo_energy(
    query_similarities: np.ndarray,
    selected_indices: List[int],
    similarity_matrix: np.ndarray,
    alpha: float = 0.05,
    K: int = 10,
    penalty: float = 1000.0
) -> float:
    """
    Compute the QUBO energy for a given selection of documents.

    Energy formula:
        E(x) = -Σᵢ S(q,dᵢ)·xᵢ                (relevance: maximize)
             + α · Σᵢ<ⱼ S(dᵢ,dⱼ)·xᵢ·xⱼ       (diversity: minimize redundancy)
             + P · (Σᵢ xᵢ - K)²              (constraint: exactly K docs)

    Args:
        query_similarities: Similarities between query and all documents
        selected_indices: Indices of selected documents
        similarity_matrix: Pairwise document similarities
        alpha: Diversity penalty weight (default: 0.05)
        K: Target number of documents
        penalty: Constraint violation penalty (default: 1000.0)

    Returns:
        Total energy value (lower is better)
    """
    # Create binary selection vector
    n_docs = len(query_similarities)
    x = np.zeros(n_docs)
    x[selected_indices] = 1

    # Relevance term (negative because we want to maximize)
    relevance_term = -np.sum(query_similarities * x)

    # Diversity term (penalty for similar documents)
    diversity_term = 0.0
    for i in selected_indices:
        for j in selected_indices:
            if i < j:
                diversity_term += alpha * similarity_matrix[i, j]

    # Cardinality constraint (penalty for not selecting exactly K docs)
    num_selected = len(selected_indices)
    constraint_term = penalty * (num_selected - K) ** 2

    return relevance_term + diversity_term + constraint_term
