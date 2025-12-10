"""
Alpha Parameter Tuning for QUBO-RAG

This script systematically searches for optimal alpha parameter values
by testing a range of candidates on hard-case scenarios and measuring:
- Intra-list similarity (lower = better diversity)
- Cluster coverage (higher = better information coverage)
- Average relevance (should stay high)
- QUBO energy (lower = better optimization)

Usage:
    python experiments/alpha_tuning.py --scenario scenario_a --k 5
    python experiments/alpha_tuning.py --all-scenarios --alpha-range 0.3 0.7 --steps 9
"""

import os
import sys
from pathlib import Path
import argparse
import json
from typing import List, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.qubo_solver import solve_diverse_retrieval_qubo
from core.retrieval_strategies import MMRRetrievalStrategy, QUBORetrievalStrategy
from core.diversity_metrics import (
    compute_intra_list_similarity,
    compute_cluster_coverage_from_filenames
)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Error: sentence-transformers not installed")
    print("Install with: pip install sentence-transformers")
    sys.exit(1)


def load_scenario_data(scenario_dir: Path) -> Tuple[Dict[str, str], Dict]:
    """Load documents and metadata for a scenario."""
    # Load documents
    documents = {}
    for filepath in scenario_dir.glob("*.txt"):
        with open(filepath, 'r', encoding='utf-8') as f:
            documents[filepath.name] = f.read()

    # Load metadata
    metadata_file = scenario_dir.parent / "scenarios_metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            all_metadata = json.load(f)
            # Find matching scenario
            for scenario_key, meta in all_metadata.items():
                scenario_name = meta.get('scenario_name', '')
                if scenario_name in scenario_dir.name:
                    return documents, meta

    return documents, {}


def run_alpha_sweep(
    query: str,
    documents: Dict[str, str],
    embeddings: Dict[str, np.ndarray],
    k: int,
    alpha_range: List[float],
    solver_preset: str = 'balanced',
    n_runs: int = 3
) -> Dict[float, Dict]:
    """
    Test different alpha values and collect metrics.

    Args:
        query: Query string
        documents: {filename: text}
        embeddings: {filename: embedding}
        k: Number of results to retrieve
        alpha_range: List of alpha values to test
        solver_preset: ORBIT solver preset
        n_runs: Number of runs per alpha (for stability)

    Returns:
        {alpha: {metrics}} dictionary
    """
    print(f"\n{'='*60}")
    print(f"Running Alpha Sweep")
    print(f"{'='*60}")
    print(f"  Query: {query}")
    print(f"  Documents: {len(documents)}")
    print(f"  k: {k}")
    print(f"  Alpha range: {alpha_range}")
    print(f"  Runs per alpha: {n_runs}")
    print(f"  Solver preset: {solver_preset}")

    # Prepare embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_emb = model.encode([query], convert_to_numpy=True)[0]

    filenames = list(documents.keys())
    candidate_embs = np.array([embeddings[fn] for fn in filenames])

    # Prepare candidate results for strategies
    candidate_results = []
    for fn, emb in zip(filenames, candidate_embs):
        # Compute query similarity
        query_norm = query_emb / np.linalg.norm(query_emb)
        emb_norm = emb / np.linalg.norm(emb)
        sim = float(np.dot(query_norm, emb_norm))

        candidate_results.append({
            'source': fn,
            'embedding': emb,
            'score': sim,
            'text': documents[fn][:200]  # First 200 chars
        })

    results = {}

    for alpha in alpha_range:
        print(f"\n  Testing alpha = {alpha:.2f}")

        alpha_metrics = {
            'intra_similarity': [],
            'cluster_coverage': [],
            'avg_relevance': [],
            'qubo_energy': [],
            'constraint_violations': []
        }

        for run in range(n_runs):
            # Run QUBO with this alpha
            strategy = QUBORetrievalStrategy(
                alpha=alpha,
                solver_preset=solver_preset
            )

            retrieved, metadata = strategy.retrieve(
                query_embedding=query_emb,
                candidate_results=candidate_results,
                k=k
            )

            # Compute metrics
            intra_sim = compute_intra_list_similarity(retrieved, embedding_key='embedding')
            coverage_info = compute_cluster_coverage_from_filenames(retrieved, total_clusters=4)

            alpha_metrics['intra_similarity'].append(intra_sim)
            alpha_metrics['cluster_coverage'].append(coverage_info['coverage_ratio'])
            alpha_metrics['avg_relevance'].append(np.mean([r.get('score', 0) for r in retrieved]))

            # Extract from metadata
            if 'solution_quality' in metadata:
                quality = metadata['solution_quality']
                alpha_metrics['qubo_energy'].append(quality['qubo_energy'])
                alpha_metrics['constraint_violations'].append(quality['constraint_violation'])

            print(f"    Run {run+1}: intra_sim={intra_sim:.3f}, "
                  f"coverage={coverage_info['coverage_ratio']:.2f}, "
                  f"clusters={coverage_info['clusters_covered']}")

        # Aggregate statistics
        results[alpha] = {
            'intra_similarity_mean': np.mean(alpha_metrics['intra_similarity']),
            'intra_similarity_std': np.std(alpha_metrics['intra_similarity']),
            'cluster_coverage_mean': np.mean(alpha_metrics['cluster_coverage']),
            'cluster_coverage_std': np.std(alpha_metrics['cluster_coverage']),
            'avg_relevance_mean': np.mean(alpha_metrics['avg_relevance']),
            'avg_relevance_std': np.std(alpha_metrics['avg_relevance']),
            'qubo_energy_mean': np.mean(alpha_metrics['qubo_energy']) if alpha_metrics['qubo_energy'] else None,
            'constraint_violations': np.mean(alpha_metrics['constraint_violations']) if alpha_metrics['constraint_violations'] else 0
        }

        print(f"    Avg: intra_sim={results[alpha]['intra_similarity_mean']:.3f} ± {results[alpha]['intra_similarity_std']:.3f}, "
              f"coverage={results[alpha]['cluster_coverage_mean']:.2f} ± {results[alpha]['cluster_coverage_std']:.2f}")

    return results


def plot_alpha_sensitivity(results: Dict[float, Dict], output_file: str):
    """Plot how metrics vary with alpha."""
    alphas = sorted(results.keys())

    intra_sims = [results[a]['intra_similarity_mean'] for a in alphas]
    intra_stds = [results[a]['intra_similarity_std'] for a in alphas]

    coverages = [results[a]['cluster_coverage_mean'] for a in alphas]
    coverage_stds = [results[a]['cluster_coverage_std'] for a in alphas]

    relevances = [results[a]['avg_relevance_mean'] for a in alphas]
    relevance_stds = [results[a]['avg_relevance_std'] for a in alphas]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Intra-list similarity (lower is better)
    axes[0].plot(alphas, intra_sims, 'o-', color='#2E86AB', linewidth=2, markersize=8)
    axes[0].fill_between(alphas,
                         [s - std for s, std in zip(intra_sims, intra_stds)],
                         [s + std for s, std in zip(intra_sims, intra_stds)],
                         alpha=0.3, color='#2E86AB')
    axes[0].set_xlabel('Alpha (Relevance Weight)', fontsize=11)
    axes[0].set_ylabel('Intra-List Similarity', fontsize=11)
    axes[0].set_title('Diversity: Lower is Better', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(min(alphas) - 0.05, max(alphas) + 0.05)

    # Plot 2: Cluster coverage (higher is better)
    axes[1].plot(alphas, coverages, 'o-', color='#A23B72', linewidth=2, markersize=8)
    axes[1].fill_between(alphas,
                         [c - std for c, std in zip(coverages, coverage_stds)],
                         [c + std for c, std in zip(coverages, coverage_stds)],
                         alpha=0.3, color='#A23B72')
    axes[1].set_xlabel('Alpha (Relevance Weight)', fontsize=11)
    axes[1].set_ylabel('Cluster Coverage Ratio', fontsize=11)
    axes[1].set_title('Information Coverage: Higher is Better', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(min(alphas) - 0.05, max(alphas) + 0.05)
    axes[1].set_ylim(0, 1.05)

    # Plot 3: Average relevance (should stay high)
    axes[2].plot(alphas, relevances, 'o-', color='#F18F01', linewidth=2, markersize=8)
    axes[2].fill_between(alphas,
                         [r - std for r, std in zip(relevances, relevance_stds)],
                         [r + std for r, std in zip(relevances, relevance_stds)],
                         alpha=0.3, color='#F18F01')
    axes[2].set_xlabel('Alpha (Relevance Weight)', fontsize=11)
    axes[2].set_ylabel('Average Relevance', fontsize=11)
    axes[2].set_title('Relevance: Should Stay High', fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(min(alphas) - 0.05, max(alphas) + 0.05)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved plot to {output_file}")
    plt.close()


def recommend_alpha(results: Dict[float, Dict]) -> Tuple[float, str]:
    """
    Recommend optimal alpha based on results.

    Strategy:
    - Maximize cluster coverage (most important)
    - Minimize intra-list similarity (diversity)
    - Keep relevance above 90% of max

    Returns:
        (recommended_alpha, reasoning)
    """
    alphas = sorted(results.keys())

    # Normalize metrics to [0, 1]
    max_relevance = max(results[a]['avg_relevance_mean'] for a in alphas)
    relevance_threshold = 0.9 * max_relevance

    # Find alphas that maintain sufficient relevance
    viable_alphas = [a for a in alphas if results[a]['avg_relevance_mean'] >= relevance_threshold]

    if not viable_alphas:
        viable_alphas = alphas  # Fallback

    # Among viable alphas, maximize coverage and minimize intra-similarity
    scores = {}
    for a in viable_alphas:
        coverage = results[a]['cluster_coverage_mean']
        diversity = 1.0 - results[a]['intra_similarity_mean']  # Invert so higher is better

        # Weighted score: 60% coverage, 40% diversity
        score = 0.6 * coverage + 0.4 * diversity
        scores[a] = score

    best_alpha = max(scores.keys(), key=lambda a: scores[a])

    reasoning = (
        f"Recommended: alpha = {best_alpha:.2f}\n"
        f"  - Cluster coverage: {results[best_alpha]['cluster_coverage_mean']:.2%}\n"
        f"  - Intra-list similarity: {results[best_alpha]['intra_similarity_mean']:.3f}\n"
        f"  - Avg relevance: {results[best_alpha]['avg_relevance_mean']:.3f}\n"
        f"  - Maintains {results[best_alpha]['avg_relevance_mean']/max_relevance:.1%} of max relevance"
    )

    return best_alpha, reasoning


def main():
    parser = argparse.ArgumentParser(description='Alpha parameter tuning for QUBO-RAG')
    parser.add_argument('--scenario', type=str, default='scenario_a_symptom_overlap',
                       help='Scenario directory name')
    parser.add_argument('--k', type=int, default=5, help='Number of results to retrieve')
    parser.add_argument('--alpha-range', nargs=2, type=float, default=[0.3, 0.7],
                       help='Alpha range: min max')
    parser.add_argument('--steps', type=int, default=9, help='Number of alpha values to test')
    parser.add_argument('--n-runs', type=int, default=3, help='Runs per alpha for stability')
    parser.add_argument('--solver-preset', type=str, default='balanced',
                       choices=['fast', 'balanced', 'quality', 'maximum'],
                       help='ORBIT solver preset')
    parser.add_argument('--output-dir', type=str, default='experiments/results',
                       help='Output directory for results')

    args = parser.parse_args()

    # Setup paths
    base_dir = Path("data/medical/raw/strategic")
    scenario_dir = base_dir / args.scenario
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not scenario_dir.exists():
        print(f"Error: Scenario directory not found: {scenario_dir}")
        print(f"Available scenarios:")
        for d in base_dir.iterdir():
            if d.is_dir():
                print(f"  - {d.name}")
        return

    # Load data
    print(f"\n{'='*60}")
    print(f"Loading scenario: {args.scenario}")
    print(f"{'='*60}")

    documents, metadata = load_scenario_data(scenario_dir)
    print(f"  Loaded {len(documents)} documents")

    query = metadata.get('query', 'Default query')
    print(f"  Query: {query}")

    # Compute embeddings
    print("\n  Computing embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = list(documents.values())
    filenames = list(documents.keys())
    embeddings_array = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = {fn: emb for fn, emb in zip(filenames, embeddings_array)}

    # Generate alpha range
    alpha_min, alpha_max = args.alpha_range
    alpha_range = np.linspace(alpha_min, alpha_max, args.steps).tolist()

    # Run sweep
    results = run_alpha_sweep(
        query=query,
        documents=documents,
        embeddings=embeddings,
        k=args.k,
        alpha_range=alpha_range,
        solver_preset=args.solver_preset,
        n_runs=args.n_runs
    )

    # Save results
    results_file = output_dir / f"alpha_sweep_{args.scenario}.json"
    with open(results_file, 'w') as f:
        # Convert float keys to strings for JSON
        json_results = {str(k): v for k, v in results.items()}
        json.dump(json_results, f, indent=2)
    print(f"\n✓ Saved results to {results_file}")

    # Plot results
    plot_file = output_dir / f"alpha_sensitivity_{args.scenario}.png"
    plot_alpha_sensitivity(results, str(plot_file))

    # Recommend alpha
    print(f"\n{'='*60}")
    print("Recommendation")
    print(f"{'='*60}")
    best_alpha, reasoning = recommend_alpha(results)
    print(reasoning)

    # Save recommendation
    rec_file = output_dir / f"recommendation_{args.scenario}.txt"
    with open(rec_file, 'w') as f:
        f.write(reasoning)
    print(f"\n✓ Saved recommendation to {rec_file}")


if __name__ == "__main__":
    main()
