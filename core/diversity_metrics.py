"""
Diversity Metrics Module

Implements metrics for evaluating retrieval diversity and quality:
- Intra-list similarity: Average pairwise similarity (lower = more diverse)
- Subtopic recall: Fraction of distinct subtopics covered
- Alpha-nDCG: Diversity-aware normalized discounted cumulative gain
- Comparison utilities: Side-by-side method evaluation

Based on:
- Clarke et al. (2008): "Subtopic Retrieval with Diversity"
- Carbonell & Goldstein (1998): "The Use of MMR and Diversity-Based Reranking"
"""

from typing import List, Dict, Any, Optional
import numpy as np


def compute_intra_list_similarity(
    results: List[Any],
    embedding_key: str = 'embedding'
) -> float:
    """
    Compute average pairwise cosine similarity within result list.

    Lower values indicate more diverse results.

    Args:
        results: List of retrieval results (must have embeddings)
        embedding_key: Key or attribute name for embeddings

    Returns:
        Average pairwise similarity in [0, 1]
    """
    if len(results) <= 1:
        return 0.0

    # Extract embeddings
    embeddings = []
    for r in results:
        if hasattr(r, embedding_key):
            embeddings.append(getattr(r, embedding_key))
        elif isinstance(r, dict) and embedding_key in r:
            embeddings.append(r[embedding_key])
        else:
            raise ValueError(f"Result missing '{embedding_key}' field")

    embeddings = np.array(embeddings)

    # Normalize vectors
    norms = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Compute pairwise similarities
    sim_matrix = np.dot(norms, norms.T)

    # Average off-diagonal elements (exclude self-similarity)
    n = len(results)
    total_sim = np.sum(sim_matrix) - n  # Subtract diagonal
    avg_sim = total_sim / (n * (n - 1))

    return float(avg_sim)


def compute_subtopic_recall(
    results: List[Any],
    ground_truth_clusters: List[List[str]],
    id_key: str = 'id'
) -> float:
    """
    Compute fraction of subtopics covered by retrieval results.

    Used for synthetic evaluation where clusters are known.

    Args:
        results: Retrieved results
        ground_truth_clusters: List of lists of chunk IDs per subtopic
        id_key: Key or attribute name for IDs

    Returns:
        Recall in [0, 1]
    """
    if not ground_truth_clusters:
        return 0.0

    # Extract IDs from results
    retrieved_ids = set()
    for r in results:
        if hasattr(r, id_key):
            retrieved_ids.add(getattr(r, id_key))
        elif isinstance(r, dict) and id_key in r:
            retrieved_ids.add(r[id_key])
        else:
            raise ValueError(f"Result missing '{id_key}' field")

    # Count covered clusters
    covered_clusters = 0
    for cluster in ground_truth_clusters:
        if any(chunk_id in retrieved_ids for chunk_id in cluster):
            covered_clusters += 1

    return covered_clusters / len(ground_truth_clusters)


def extract_disease_from_filename(filename: str) -> str:
    """
    Extract disease cluster name from filename.

    Examples:
        'chronic_fatigue_variant_1.txt' → 'chronic_fatigue'
        'lupus_doc_3.txt' → 'lupus'
        'ra_symptoms_2.txt' → 'ra'
        'cfs_doc_1.txt' → 'cfs'

    Args:
        filename: Document filename

    Returns:
        Disease cluster name
    """
    # Remove extension
    name = filename.replace('.txt', '')

    # Common patterns
    # Pattern 1: disease_variant_N or disease_doc_N
    for suffix in ['_variant', '_doc', '_symptoms', '_diagnosis', '_treatment',
                   '_complications', '_risk', '_prognosis', '_clinical', '_multi',
                   '_extra']:
        if suffix in name:
            parts = name.split(suffix)
            return parts[0]

    # Pattern 2: disease_N (just disease name and number)
    parts = name.split('_')
    if len(parts) >= 2 and parts[-1].isdigit():
        return '_'.join(parts[:-1])

    # Pattern 3: Return whole name without numbers
    return name.rstrip('0123456789_')


def compute_cluster_coverage_from_filenames(
    results: List[Any],
    total_clusters: int = 4,
    source_key: str = 'source'
) -> Dict[str, Any]:
    """
    Extract disease clusters from filenames and compute coverage metrics.

    This function works with the strategic synthetic data where cluster
    membership is encoded in filenames.

    Args:
        results: Retrieved results
        total_clusters: Total number of clusters in dataset (default: 4)
        source_key: Key or attribute name for source filename

    Returns:
        Dictionary with:
            - clusters_covered: List of unique disease names
            - coverage_count: Number of clusters covered
            - coverage_ratio: Fraction of clusters covered
            - cluster_distribution: Dict mapping disease to count
    """
    cluster_names = []

    for r in results:
        # Extract source filename
        if hasattr(r, source_key):
            source = getattr(r, source_key)
        elif isinstance(r, dict) and source_key in r:
            source = r[source_key]
        else:
            continue

        # Extract disease name
        disease = extract_disease_from_filename(source)
        cluster_names.append(disease)

    # Unique clusters
    unique_clusters = list(set(cluster_names))

    # Cluster distribution
    cluster_counts = {disease: cluster_names.count(disease) for disease in unique_clusters}

    return {
        'clusters_covered': unique_clusters,
        'coverage_count': len(unique_clusters),
        'coverage_ratio': len(unique_clusters) / total_clusters if total_clusters > 0 else 0.0,
        'cluster_distribution': cluster_counts
    }


def compute_alpha_ndcg(
    results: List[Any],
    ground_truth_relevance: Dict[str, float],
    ground_truth_clusters: List[List[str]],
    alpha: float = 0.5,
    id_key: str = 'id'
) -> float:
    """
    Compute α-nDCG: Diversity-aware normalized discounted cumulative gain.

    Based on Clarke et al. (2008): "Novelty and Diversity in Information Retrieval Evaluation"

    Rewards both relevance and subtopic coverage with diminishing returns
    for repeated coverage of the same subtopic.

    Args:
        results: Retrieved results (in rank order)
        ground_truth_relevance: {chunk_id: relevance_score}
        ground_truth_clusters: Subtopic clusters
        alpha: Diminishing return parameter (default 0.5)
        id_key: Key or attribute name for IDs

    Returns:
        α-nDCG score in [0, 1]
    """
    if not results:
        return 0.0

    # Build cluster membership map
    chunk_to_cluster = {}
    for cluster_idx, cluster in enumerate(ground_truth_clusters):
        for chunk_id in cluster:
            chunk_to_cluster[chunk_id] = cluster_idx

    # Compute α-DCG
    cluster_coverage = {}  # {cluster_id: count}
    dcg = 0.0

    for rank, result in enumerate(results, start=1):
        # Extract ID
        if hasattr(result, id_key):
            chunk_id = getattr(result, id_key)
        elif isinstance(result, dict) and id_key in r:
            chunk_id = result[id_key]
        else:
            continue

        relevance = ground_truth_relevance.get(chunk_id, 0.0)
        cluster_id = chunk_to_cluster.get(chunk_id, -1)

        if cluster_id >= 0:
            coverage = cluster_coverage.get(cluster_id, 0)
            cluster_coverage[cluster_id] = coverage + 1

            # Diminishing return for repeated cluster
            gain = relevance * (1 - alpha) ** coverage
        else:
            gain = relevance

        # Discounted gain
        dcg += gain / np.log2(rank + 1)

    # Compute ideal α-DCG (sort by relevance, distributed across clusters)
    # For simplicity, use standard nDCG normalization
    ideal_scores = sorted(ground_truth_relevance.values(), reverse=True)[:len(results)]
    ideal_dcg = sum(score / np.log2(rank + 1) for rank, score in enumerate(ideal_scores, start=1))

    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def compute_diversity_metrics(
    results: List[Any],
    ground_truth_data: Optional[Dict[str, Any]] = None,
    embedding_key: str = 'embedding',
    id_key: str = 'id'
) -> Dict[str, float]:
    """
    Compute all diversity metrics for a result set.

    Args:
        results: Retrieved results
        ground_truth_data: Optional dict with 'relevance', 'clusters' keys
        embedding_key: Key for embeddings
        id_key: Key for IDs

    Returns:
        Dict with metric names and values
    """
    metrics = {
        'num_results': len(results)
    }

    # Always compute intra-list similarity if embeddings available
    try:
        metrics['intra_list_similarity'] = compute_intra_list_similarity(
            results, embedding_key=embedding_key
        )
    except (ValueError, KeyError):
        pass

    # Compute score statistics if available
    try:
        scores = []
        for r in results:
            if hasattr(r, 'score'):
                scores.append(r.score)
            elif isinstance(r, dict) and 'score' in r:
                scores.append(r['score'])

        if scores:
            metrics['avg_score'] = np.mean(scores)
            metrics['min_score'] = np.min(scores)
            metrics['max_score'] = np.max(scores)
            metrics['std_score'] = np.std(scores)
    except (ValueError, KeyError):
        pass

    # Compute ground truth metrics if provided
    if ground_truth_data:
        if 'clusters' in ground_truth_data:
            metrics['subtopic_recall'] = compute_subtopic_recall(
                results, ground_truth_data['clusters'], id_key=id_key
            )

        if 'relevance' in ground_truth_data and 'clusters' in ground_truth_data:
            metrics['alpha_ndcg'] = compute_alpha_ndcg(
                results,
                ground_truth_data['relevance'],
                ground_truth_data['clusters'],
                id_key=id_key
            )

    return metrics


def compare_retrieval_methods(
    results_dict: Dict[str, List[Any]],
    ground_truth_data: Optional[Dict[str, Any]] = None,
    embedding_key: str = 'embedding',
    id_key: str = 'id'
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple retrieval methods side-by-side.

    Args:
        results_dict: {method_name: results_list}
        ground_truth_data: Optional ground truth
        embedding_key: Key for embeddings
        id_key: Key for IDs

    Returns:
        {method_name: {metric_name: value}}

    Example:
        results = {
            'Naive': naive_results,
            'MMR': mmr_results,
            'QUBO': qubo_results
        }
        comparison = compare_retrieval_methods(results, ground_truth)
        print(f"Naive ILS: {comparison['Naive']['intra_list_similarity']:.3f}")
        print(f"QUBO ILS: {comparison['QUBO']['intra_list_similarity']:.3f}")
    """
    comparison = {}

    for method_name, results in results_dict.items():
        comparison[method_name] = compute_diversity_metrics(
            results, ground_truth_data, embedding_key=embedding_key, id_key=id_key
        )

    return comparison


def print_comparison_table(comparison: Dict[str, Dict[str, float]]):
    """
    Print a formatted comparison table.

    Args:
        comparison: Output from compare_retrieval_methods()
    """
    if not comparison:
        print("No results to compare")
        return

    # Get all metrics
    all_metrics = set()
    for metrics in comparison.values():
        all_metrics.update(metrics.keys())

    all_metrics = sorted(all_metrics)

    # Print header
    methods = list(comparison.keys())
    header = f"{'Metric':<25} " + " ".join(f"{m:>15}" for m in methods)
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    # Print each metric
    for metric in all_metrics:
        values = []
        for method in methods:
            value = comparison[method].get(metric, float('nan'))
            if isinstance(value, (int, np.integer)):
                values.append(f"{value:>15d}")
            else:
                values.append(f"{value:>15.4f}")

        print(f"{metric:<25} " + " ".join(values))

    print("=" * len(header))

    # Print interpretation
    print("\nInterpretation:")
    print("- Intra-list similarity: Lower = more diverse")
    print("- Subtopic recall: Higher = better coverage")
    print("- Alpha-nDCG: Higher = better relevance + diversity balance")
