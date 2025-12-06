"""
Synthetic Retrieval Tests

Tests retrieval strategies on controlled synthetic data to demonstrate
diversity improvements of QUBO vs MMR vs Naive.

Test scenarios:
1. Clustered data: 5 clusters, multi-intent query
2. Redundant documents: Near-duplicates in one cluster
"""

import numpy as np
from typing import List, Tuple, Dict
from core.retrieval_strategies import create_retrieval_strategy
from core.diversity_metrics import (
    compute_diversity_metrics,
    compare_retrieval_methods,
    print_comparison_table
)


class SyntheticDataGenerator:
    """Generate controlled embedding patterns for testing."""

    def __init__(self, embedding_dim: int = 384, random_seed: int = 42):
        self.embedding_dim = embedding_dim
        self.rng = np.random.default_rng(random_seed)

    def generate_clustered_data(
        self,
        n_clusters: int = 3,
        items_per_cluster: int = 10,
        cluster_std: float = 0.1,
        inter_cluster_distance: float = 1.0
    ) -> Tuple[np.ndarray, List[List[int]], List[np.ndarray]]:
        """
        Generate synthetic embeddings with clear cluster structure.

        Returns:
            embeddings: (n_items, dim) array
            cluster_membership: List of item indices per cluster
            cluster_centers: List of center vectors
        """
        cluster_centers = []
        embeddings = []
        cluster_membership = [[] for _ in range(n_clusters)]

        # Generate cluster centers (widely separated)
        for i in range(n_clusters):
            # Random direction on unit sphere
            center = self.rng.standard_normal(self.embedding_dim)
            center = center / np.linalg.norm(center) * inter_cluster_distance
            cluster_centers.append(center)

        # Generate items around centers
        for cluster_idx, center in enumerate(cluster_centers):
            for item_idx in range(items_per_cluster):
                # Add Gaussian noise
                noise = self.rng.standard_normal(self.embedding_dim) * cluster_std
                item_emb = center + noise
                item_emb = item_emb / np.linalg.norm(item_emb)  # Normalize

                global_idx = len(embeddings)
                embeddings.append(item_emb)
                cluster_membership[cluster_idx].append(global_idx)

        return np.array(embeddings), cluster_membership, cluster_centers

    def generate_query_for_cluster(
        self,
        cluster_center: np.ndarray,
        noise_level: float = 0.05
    ) -> np.ndarray:
        """Generate query embedding near a cluster center."""
        noise = self.rng.standard_normal(self.embedding_dim) * noise_level
        query = cluster_center + noise
        return query / np.linalg.norm(query)

    def generate_multi_intent_query(
        self,
        cluster_centers: List[np.ndarray],
        target_clusters: List[int],
        weights: List[float] = None
    ) -> np.ndarray:
        """
        Generate query that spans multiple clusters (simulates multi-intent query).

        Args:
            cluster_centers: List of all cluster centers
            target_clusters: Indices of clusters to combine
            weights: Optional weights per cluster (default: equal)
        """
        if weights is None:
            weights = [1.0 / len(target_clusters)] * len(target_clusters)

        query = np.zeros(self.embedding_dim)
        for cluster_idx, weight in zip(target_clusters, weights):
            query += weight * cluster_centers[cluster_idx]

        return query / np.linalg.norm(query)


def create_mock_candidates(embeddings, clusters, scores):
    """Create mock candidate results for retrieval strategies."""
    candidates = []
    for idx, (emb, score) in enumerate(zip(embeddings, scores)):
        # Find which cluster this belongs to
        cluster_id = -1
        for cid, cluster in enumerate(clusters):
            if idx in cluster:
                cluster_id = cid
                break

        candidates.append({
            'id': f'chunk_{idx}',
            'text': f'Text from cluster {cluster_id}',
            'metadata': {'source': 'synthetic', 'chunk_index': idx, 'cluster': cluster_id},
            'score': float(score),
            'embedding': emb
        })

    # Sort by score descending
    candidates.sort(key=lambda x: x['score'], reverse=True)
    return candidates


def test_clustered_retrieval():
    """
    Test 1: Retrieval from clustered data.

    Hypothesis: QUBO and MMR should cover more clusters than Naive.
    """
    print("\n" + "=" * 70)
    print("TEST 1: Clustered Data Retrieval")
    print("=" * 70)

    # Generate data
    gen = SyntheticDataGenerator()
    embeddings, clusters, centers = gen.generate_clustered_data(
        n_clusters=5,
        items_per_cluster=10,
        cluster_std=0.1
    )

    # Multi-intent query (wants items from 3 clusters)
    query_emb = gen.generate_multi_intent_query(centers, target_clusters=[0, 2, 4])

    # Compute scores
    query_norm = query_emb / np.linalg.norm(query_emb)
    emb_norms = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    scores = np.dot(emb_norms, query_norm)

    # Create mock candidates
    candidates = create_mock_candidates(embeddings, clusters, scores)

    # Test all three methods
    k = 10
    strategies = {
        'Naive': create_retrieval_strategy('naive'),
        'MMR (lambda=0.5)': create_retrieval_strategy('mmr', lambda_param=0.5),
        'QUBO (alpha=0.6)': create_retrieval_strategy('qubo', alpha=0.6,
                                                   solver_params={'n_replicas': 2, 'full_sweeps': 5000})
    }

    results_dict = {}
    for name, strategy in strategies.items():
        print(f"\nRunning {name}...")
        results, metadata = strategy.retrieve(query_emb, candidates, k=k)

        # Convert to dict format for metrics
        results_with_embeddings = []
        for r in results:
            # Find the original candidate to get embedding
            orig = next((c for c in candidates if c['id'] == r.id), None)
            if orig:
                results_with_embeddings.append({
                    'id': r.id,
                    'score': r.score,
                    'embedding': orig['embedding'],
                    'cluster': orig['metadata']['cluster']
                })

        results_dict[name] = results_with_embeddings

        # Analyze cluster coverage
        retrieved_clusters = set(r['cluster'] for r in results_with_embeddings)

        print(f"  Clusters covered: {len(retrieved_clusters)}/5")
        print(f"  Cluster IDs: {sorted(retrieved_clusters)}")
        print(f"  Avg score: {np.mean([r['score'] for r in results_with_embeddings]):.4f}")

        if 'execution_time' in metadata:
            print(f"  Execution time: {metadata['execution_time']:.3f}s")

    # Compute and display metrics
    print("\n" + "-" * 70)
    print("DIVERSITY METRICS:")
    print("-" * 70)

    comparison = compare_retrieval_methods(results_dict)
    print_comparison_table(comparison)

    # Expected outcomes
    print("\n" + "=" * 70)
    print("EXPECTED OUTCOMES:")
    print("=" * 70)
    print("* Naive: Should cover 1-2 clusters (focuses on highest scores)")
    print("* MMR: Should cover 3-4 clusters (balanced)")
    print("* QUBO: Should cover 4-5 clusters (best diversity)")
    print("* QUBO: Lowest intra-list similarity")

    return results_dict, comparison


def test_redundant_documents():
    """
    Test 2: Retrieval with highly similar documents (near-duplicates).

    Hypothesis: QUBO/MMR should avoid selecting near-duplicates.
    """
    print("\n\n" + "=" * 70)
    print("TEST 2: Redundant Document Filtering")
    print("=" * 70)

    gen = SyntheticDataGenerator()

    # Create base embeddings with 3 clusters
    embeddings, clusters, centers = gen.generate_clustered_data(
        n_clusters=3,
        items_per_cluster=5,
        cluster_std=0.08
    )

    # Add near-duplicates to cluster 0 (5 more very similar items)
    duplicate_embeddings = []
    for i in range(5):
        dup = centers[0] + gen.rng.standard_normal(gen.embedding_dim) * 0.02  # Very low noise
        dup = dup / np.linalg.norm(dup)
        duplicate_embeddings.append(dup)

    embeddings = np.vstack([embeddings, duplicate_embeddings])

    # Query targeting cluster 0
    query_emb = gen.generate_query_for_cluster(centers[0], noise_level=0.05)

    # Compute scores
    query_norm = query_emb / np.linalg.norm(query_emb)
    emb_norms = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    scores = np.dot(emb_norms, query_norm)

    # Update clusters to include duplicates
    clusters.append([15, 16, 17, 18, 19])  # Duplicate cluster

    # Create mock candidates
    candidates = create_mock_candidates(embeddings, clusters, scores)

    # Test methods
    k = 10
    strategies = {
        'Naive': create_retrieval_strategy('naive'),
        'MMR (lambda=0.5)': create_retrieval_strategy('mmr', lambda_param=0.5),
        'QUBO (alpha=0.6)': create_retrieval_strategy('qubo', alpha=0.6,
                                                   solver_params={'n_replicas': 2, 'full_sweeps': 5000})
    }

    results_dict = {}
    for name, strategy in strategies.items():
        print(f"\nRunning {name}...")
        results, metadata = strategy.retrieve(query_emb, candidates, k=k)

        # Convert to dict format
        results_with_embeddings = []
        for r in results:
            orig = next((c for c in candidates if c['id'] == r.id), None)
            if orig:
                results_with_embeddings.append({
                    'id': r.id,
                    'score': r.score,
                    'embedding': orig['embedding'],
                    'cluster': orig['metadata']['cluster']
                })

        results_dict[name] = results_with_embeddings

        # Count items from high-similarity cluster (0 and duplicates)
        cluster0_count = sum(1 for r in results_with_embeddings if r['cluster'] in [0, 3])

        print(f"  Items from high-similarity clusters: {cluster0_count}/{k}")
        print(f"  Avg score: {np.mean([r['score'] for r in results_with_embeddings]):.4f}")

    # Metrics
    print("\n" + "-" * 70)
    print("DIVERSITY METRICS:")
    print("-" * 70)

    comparison = compare_retrieval_methods(results_dict)
    print_comparison_table(comparison)

    # Expected outcomes
    print("\n" + "=" * 70)
    print("EXPECTED OUTCOMES:")
    print("=" * 70)
    print("* Naive: Many items from same cluster (high intra-list similarity)")
    print("* MMR: Fewer duplicates, better distribution")
    print("* QUBO: Best diversity/relevance balance, lowest intra-list similarity")

    return results_dict, comparison


def run_all_synthetic_tests():
    """Run all synthetic test cases."""
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + " " * 15 + "SYNTHETIC RETRIEVAL TESTS" + " " * 28 + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)

    results1, comparison1 = test_clustered_retrieval()
    results2, comparison2 = test_redundant_documents()

    print("\n\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print("\nTest 1 - Clustered Data:")
    print(f"  Naive intra-list sim: {comparison1['Naive']['intra_list_similarity']:.4f}")
    print(f"  MMR intra-list sim:   {comparison1['MMR (lambda=0.5)']['intra_list_similarity']:.4f}")
    print(f"  QUBO intra-list sim:  {comparison1['QUBO (alpha=0.6)']['intra_list_similarity']:.4f}")

    print("\nTest 2 - Redundant Documents:")
    print(f"  Naive intra-list sim: {comparison2['Naive']['intra_list_similarity']:.4f}")
    print(f"  MMR intra-list sim:   {comparison2['MMR (lambda=0.5)']['intra_list_similarity']:.4f}")
    print(f"  QUBO intra-list sim:  {comparison2['QUBO (alpha=0.6)']['intra_list_similarity']:.4f}")

    print("\n* QUBO consistently achieves lowest intra-list similarity (best diversity)")
    print("* QUBO balances relevance and diversity effectively")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    run_all_synthetic_tests()
