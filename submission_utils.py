"""
Submission Utilities for Quantum-RAG Competition
Self-contained helper functions for the demonstration notebook.
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import gurobipy as gp
from gurobipy import GRB


# ============================================================================
# DATA LOADING
# ============================================================================

def load_wikipedia_dataset(data_dir: str = './data/wikipedia') -> Tuple[List[Dict], Dict[str, np.ndarray]]:
    """
    Load Wikipedia dataset with chunks and embeddings.

    Returns:
        chunks: List of chunk dictionaries
        embeddings: Dictionary mapping chunk_id -> embedding vector
    """
    data_path = Path(data_dir)

    # Load chunks from JSONL
    chunks = []
    chunks_file = data_path / 'checkpoints' / 'chunks.jsonl'
    with open(chunks_file, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))

    # Load embeddings from NPZ
    embeddings_file = data_path / 'checkpoints' / 'embeddings.npz'
    embeddings_npz = np.load(embeddings_file)
    embeddings = {key: embeddings_npz[key] for key in embeddings_npz.keys()}

    return chunks, embeddings


def filter_chunks_by_prompt(chunks: List[Dict],
                            prompt_id: int,
                            redundancy_level: int = 0) -> Tuple[List[Dict], set]:
    """
    Filter chunks for a specific prompt at a given redundancy level.

    Args:
        chunks: All chunks
        prompt_id: The prompt ID to filter for
        redundancy_level: How many redundant copies to include (0 = base only)

    Returns:
        candidate_chunks: Filtered chunks (gold_base + redundant + noise)
        gold_aspects: Set of aspect IDs that should be retrieved
    """
    candidates = []
    gold_aspects = set()

    for chunk in chunks:
        if chunk.get('prompt_id') != prompt_id:
            continue

        chunk_type = chunk.get('chunk_type', '')

        if chunk_type == 'gold_base':
            # Always include base gold chunks
            candidates.append(chunk)
            aspect_id = chunk.get('aspect_id', -1)
            if aspect_id >= 0:
                gold_aspects.add(aspect_id)

        elif chunk_type == 'gold_redundant':
            # Include redundant chunks up to redundancy_level
            redundancy_idx = chunk.get('redundancy_index', -1)
            if redundancy_idx < redundancy_level:
                candidates.append(chunk)

        elif chunk_type == 'noise':
            # Always include noise
            candidates.append(chunk)

    return candidates, gold_aspects


def get_prompt_embedding(chunks: List[Dict],
                         embeddings: Dict,
                         prompt_id: int) -> Optional[np.ndarray]:
    """Get the embedding for a specific prompt."""
    prompt_chunks = [c for c in chunks
                    if c.get('chunk_type') == 'prompt'
                    and c.get('prompt_id') == prompt_id]

    if not prompt_chunks:
        return None

    chunk_id = prompt_chunks[0]['chunk_id']
    return embeddings.get(chunk_id)


# ============================================================================
# SIMILARITY COMPUTATION
# ============================================================================

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


def compute_similarity_matrix(embeddings_list: List[np.ndarray]) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix.

    Args:
        embeddings_list: List of embedding vectors

    Returns:
        similarity_matrix: n x n matrix where [i,j] = similarity(i, j)
    """
    n = len(embeddings_list)
    S = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            S[i, j] = cosine_similarity(embeddings_list[i], embeddings_list[j])

    return S


# ============================================================================
# TOP-K RETRIEVAL (BASELINE)
# ============================================================================

def retrieve_topk(query_embedding: np.ndarray,
                  candidate_chunks: List[Dict],
                  candidate_embeddings: Dict,
                  k: int = 5) -> List[Dict]:
    """
    Standard Top-K retrieval: select k chunks with highest similarity.

    Args:
        query_embedding: Query vector
        candidate_chunks: List of candidate chunks
        candidate_embeddings: Dictionary of embeddings
        k: Number of chunks to retrieve

    Returns:
        selected_chunks: Top-k chunks by similarity
    """
    # Compute similarities
    scored_chunks = []
    for chunk in candidate_chunks:
        chunk_id = chunk['chunk_id']
        chunk_emb = candidate_embeddings.get(chunk_id)

        if chunk_emb is None:
            continue

        similarity = cosine_similarity(query_embedding, chunk_emb)
        scored_chunks.append({
            'chunk': chunk,
            'score': similarity
        })

    # Sort by similarity (descending)
    scored_chunks.sort(key=lambda x: x['score'], reverse=True)

    # Return top-k
    return [item['chunk'] for item in scored_chunks[:k]]


# ============================================================================
# QUBO RETRIEVAL (OURS)
# ============================================================================

def retrieve_qubo_gurobi(query_embedding: np.ndarray,
                        candidate_chunks: List[Dict],
                        candidate_embeddings: Dict,
                        k: int = 5,
                        alpha: float = 0.20,
                        penalty: float = 1000.0,
                        verbose: bool = False) -> Tuple[List[Dict], Dict]:
    """
    QUBO-based retrieval using Gurobi optimizer.

    Energy Function:
        E = -Σ(relevance_i * x_i) + α * ΣΣ(similarity_ij * x_i * x_j) + P * (Σx_i - k)²

    Where:
        - relevance_i: cosine similarity between query and chunk i
        - similarity_ij: cosine similarity between chunks i and j
        - x_i: binary variable (1 = selected, 0 = not selected)
        - α: diversity penalty weight (higher = more diversity emphasis)
        - P: cardinality penalty (enforces exactly k chunks selected)
        - k: desired number of chunks

    Args:
        query_embedding: Query vector
        candidate_chunks: List of candidate chunks
        candidate_embeddings: Dictionary of embeddings
        k: Number of chunks to retrieve
        alpha: Diversity penalty weight (higher = more diversity)
        penalty: Cardinality constraint penalty
        verbose: Print Gurobi output

    Returns:
        selected_chunks: Optimally selected chunks
        metadata: Solver metadata (objective value, time, etc.)
    """
    n = len(candidate_chunks)

    # Prepare embeddings and compute similarities
    chunk_ids = []
    chunk_embs = []
    relevance_scores = []

    for chunk in candidate_chunks:
        chunk_id = chunk['chunk_id']
        chunk_emb = candidate_embeddings.get(chunk_id)

        if chunk_emb is None:
            continue

        chunk_ids.append(chunk_id)
        chunk_embs.append(chunk_emb)
        relevance_scores.append(cosine_similarity(query_embedding, chunk_emb))

    n = len(chunk_ids)
    relevance_scores = np.array(relevance_scores)

    # Compute pairwise similarity matrix
    similarity_matrix = compute_similarity_matrix(chunk_embs)

    # Create Gurobi model
    model = gp.Model("QUBO_Retrieval")
    if not verbose:
        model.setParam('OutputFlag', 0)

    # Binary decision variables
    x = model.addVars(n, vtype=GRB.BINARY, name="x")

    # Objective function components (matching markdown formulation)
    # 1. Relevance term: maximize Σ(relevance_i * x_i)
    #    Negative sign for minimization (no alpha multiplier)
    relevance_term = -gp.quicksum(relevance_scores[i] * x[i] for i in range(n))

    # 2. Diversity term: minimize α * ΣΣ(similarity_ij * x_i * x_j)
    #    Higher similarity = less diverse, so we penalize it
    #    Only alpha multiplier (not 1-alpha)
    diversity_term = alpha * gp.quicksum(
        similarity_matrix[i, j] * x[i] * x[j]
        for i in range(n)
        for j in range(i+1, n)
    )

    # 3. Cardinality penalty: P * (Σx_i - k)²
    sum_x = gp.quicksum(x[i] for i in range(n))
    cardinality_penalty = penalty * (sum_x - k) * (sum_x - k)

    # Total objective: E = -r^T x + α x^T S x + P(Σx - k)^2
    objective = relevance_term + diversity_term + cardinality_penalty

    model.setObjective(objective, GRB.MINIMIZE)

    # Solve
    model.optimize()

    # Extract solution
    selected_indices = [i for i in range(n) if x[i].X > 0.5]
    selected_chunks = [candidate_chunks[i] for i in selected_indices
                      if chunk_ids[i] == candidate_chunks[i]['chunk_id']]

    metadata = {
        'objective_value': model.ObjVal,
        'solve_time': model.Runtime,
        'num_selected': len(selected_indices),
        'alpha': alpha,
        'penalty': penalty,
        'avg_relevance': float(np.mean([relevance_scores[i] for i in selected_indices])) if selected_indices else 0.0
    }

    return selected_chunks, metadata


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def compute_aspect_recall(retrieved_chunks: List[Dict],
                         gold_aspects: set) -> Tuple[float, int]:
    """
    Compute aspect recall: how many distinct gold aspects were retrieved?

    Args:
        retrieved_chunks: List of retrieved chunks
        gold_aspects: Set of aspect IDs that should be retrieved

    Returns:
        recall_percent: Percentage of gold aspects retrieved (0-100)
        num_retrieved: Count of distinct gold aspects retrieved
    """
    retrieved_aspects = set()

    for chunk in retrieved_chunks:
        aspect_id = chunk.get('aspect_id', -1)
        if aspect_id >= 0:  # Valid aspect (not noise)
            retrieved_aspects.add(aspect_id)

    num_gold = len(gold_aspects)
    if num_gold == 0:
        return 0.0, 0

    num_retrieved = len(retrieved_aspects & gold_aspects)
    recall = 100.0 * num_retrieved / num_gold

    return recall, num_retrieved


def compute_intra_list_similarity(retrieved_chunks: List[Dict],
                                  embeddings: Dict) -> float:
    """
    Compute average pairwise similarity within retrieved set.
    Lower = more diverse.

    Args:
        retrieved_chunks: List of retrieved chunks
        embeddings: Dictionary of embeddings

    Returns:
        avg_similarity: Average pairwise cosine similarity
    """
    chunk_embs = []
    for chunk in retrieved_chunks:
        chunk_id = chunk['chunk_id']
        emb = embeddings.get(chunk_id)
        if emb is not None:
            chunk_embs.append(emb)

    if len(chunk_embs) < 2:
        return 0.0

    similarities = []
    for i in range(len(chunk_embs)):
        for j in range(i+1, len(chunk_embs)):
            sim = cosine_similarity(chunk_embs[i], chunk_embs[j])
            similarities.append(sim)

    return float(np.mean(similarities))


# ============================================================================
# VISUALIZATION HELPERS
# ============================================================================

def print_retrieval_results(retrieved_chunks: List[Dict],
                           gold_aspects: set,
                           method_name: str = "Retrieval",
                           show_text_preview: bool = True):
    """
    Print retrieval results in a readable format.

    Args:
        retrieved_chunks: List of retrieved chunks
        gold_aspects: Set of gold aspect IDs
        method_name: Name to display for this method
        show_text_preview: Whether to show text previews
    """
    print(f"\n{'='*80}")
    print(f"{method_name.upper()} RESULTS")
    print(f"{'='*80}")

    for i, chunk in enumerate(retrieved_chunks, 1):
        chunk_type = chunk.get('chunk_type', 'unknown')
        aspect_id = chunk.get('aspect_id', -1)
        aspect_name = chunk.get('aspect_name', 'N/A')
        redundancy_idx = chunk.get('redundancy_index', -1)

        # Emoji for chunk type
        if chunk_type == 'gold_base':
            emoji = '✓'
            type_str = 'GOLD BASE'
        elif chunk_type == 'gold_redundant':
            emoji = '↻'
            type_str = f'REDUNDANT #{redundancy_idx}'
        elif chunk_type == 'noise':
            emoji = '✗'
            type_str = 'NOISE'
        else:
            emoji = '?'
            type_str = chunk_type.upper()

        # Check if this aspect is in gold set
        is_gold = aspect_id in gold_aspects if aspect_id >= 0 else False
        gold_marker = '⭐' if is_gold else ''

        print(f"\n[{i}] {emoji} Aspect {aspect_id}: {aspect_name} | {type_str} {gold_marker}")

        if show_text_preview:
            text = chunk.get('text', '')
            preview = text[:120] + '...' if len(text) > 120 else text
            print(f"    \"{preview}\"")

    # Compute metrics
    recall, num_retrieved = compute_aspect_recall(retrieved_chunks, gold_aspects)

    print(f"\n{'-'*80}")
    print(f"Aspect Recall: {recall:.0f}% ({num_retrieved}/{len(gold_aspects)} gold aspects retrieved)")
    print(f"Total Retrieved: {len(retrieved_chunks)} chunks")
    print(f"{'='*80}\n")


def print_comparison_table(topk_results: Dict,
                          qubo_results: Dict,
                          gold_aspects: set,
                          embeddings: Dict):
    """
    Print side-by-side comparison of Top-K vs QUBO.

    Args:
        topk_results: Top-K retrieved chunks and metadata
        qubo_results: QUBO retrieved chunks and metadata
        gold_aspects: Gold aspect IDs
        embeddings: Embeddings dictionary
    """
    topk_chunks = topk_results['chunks']
    qubo_chunks = qubo_results['chunks']

    topk_recall, topk_count = compute_aspect_recall(topk_chunks, gold_aspects)
    qubo_recall, qubo_count = compute_aspect_recall(qubo_chunks, gold_aspects)

    topk_diversity = compute_intra_list_similarity(topk_chunks, embeddings)
    qubo_diversity = compute_intra_list_similarity(qubo_chunks, embeddings)

    print("\n" + "="*80)
    print("COMPARISON: TOP-K vs QUBO-RAG")
    print("="*80)
    print(f"{'Metric':<40} {'Top-K':>15} {'QUBO-RAG':>15}")
    print("-"*80)
    print(f"{'Aspect Recall (%)':<40} {topk_recall:>15.1f} {qubo_recall:>15.1f}")
    print(f"{'Aspects Retrieved (out of ' + str(len(gold_aspects)) + ')':<40} {topk_count:>15} {qubo_count:>15}")
    print(f"{'Intra-List Similarity (lower=diverse)':<40} {topk_diversity:>15.3f} {qubo_diversity:>15.3f}")
    print(f"{'Avg Relevance Score':<40} {topk_results.get('avg_relevance', 0):>15.3f} {qubo_results.get('avg_relevance', 0):>15.3f}")
    print(f"{'Chunks Retrieved':<40} {len(topk_chunks):>15} {len(qubo_chunks):>15}")

    if 'solve_time' in qubo_results:
        print(f"{'QUBO Solve Time (s)':<40} {'-':>15} {qubo_results['solve_time']:>15.3f}")

    print("="*80)

    # Improvement
    if topk_recall > 0:
        improvement = ((qubo_recall - topk_recall) / topk_recall) * 100
        print(f"\n📊 QUBO improves aspect recall by {improvement:+.1f}%")

    diversity_improvement = ((topk_diversity - qubo_diversity) / topk_diversity) * 100 if topk_diversity > 0 else 0
    print(f"📊 QUBO reduces redundancy by {diversity_improvement:+.1f}%")
    print()
