"""
Experiment 0.1: QUBO Energy-Quality Correlation

Purpose: Validate that QUBO energy mathematically correlates with retrieval quality.

Hypothesis: Lower QUBO energy directly predicts higher percentage of distinct gold facts
retrieved. The formulation has theoretical grounding, not arbitrary filtering.

Method:
1. For small candidate pools (N=30), enumerate all possible K=5 subsets via brute force
2. For each subset, compute:
   - QUBO energy E = -α·Σ(relevance) + β·Σ(redundancy)
   - Gold fact percentage (percentage of distinct gold facts retrieved)
3. Plot: Gold Fact % (x-axis) vs Normalized QUBO Energy (y-axis) with error bars

Dataset: 100 prompts from Wikipedia dataset, redundancy level 0 only (5 gold base + 25 noise)

Settings:
- N = 30 (candidate pool size: 5 gold + 25 noise)
- K = 5 (context window)
- Total subsets per prompt: C(30,5) = 142,506
- Energies normalized to [0,1] per prompt
- Alpha = 0.05 (diversity weight)

Success Metric: Pearson correlation r < -0.7 (strong negative correlation)
"""

import sys
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
from itertools import combinations
from tqdm import tqdm
from scipy.stats import pearsonr
from collections import defaultdict

from core.qubo_solver import QUBOProblem


def load_wikipedia_data():
    """Load Wikipedia dataset (chunks, embeddings, id_mapping)."""
    data_dir = project_root / 'data' / 'wikipedia'

    # Load chunks
    chunks = []
    with open(data_dir / 'checkpoints' / 'chunks.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))

    # Load embeddings (NPZ keyed by chunk_id)
    embeddings_npz = np.load(data_dir / 'checkpoints' / 'embeddings.npz')
    embeddings_dict = {key: embeddings_npz[key] for key in embeddings_npz.keys()}

    print(f"Loaded {len(chunks)} chunks")
    print(f"Loaded {len(embeddings_dict)} embeddings")
    print(f"Embedding dimension: {list(embeddings_dict.values())[0].shape[0]}")

    return chunks, embeddings_dict


def filter_chunks_for_prompt(chunks, prompt_id, redundancy_level=0):
    """
    Filter chunks for a specific prompt at a given redundancy level.

    Args:
        chunks: List of all chunks
        prompt_id: Prompt ID to filter for
        redundancy_level: Maximum redundancy level to include (0 = base only)

    Returns:
        gold_chunks: List of gold chunks (base + redundant up to level)
        noise_chunks: List of noise chunks
    """
    gold_chunks = []
    noise_chunks = []

    for chunk in chunks:
        if chunk.get('prompt_id') != prompt_id:
            continue

        chunk_type = chunk.get('chunk_type', '')

        if chunk_type == 'gold_base':
            gold_chunks.append(chunk)
        elif chunk_type == 'gold_redundant':
            redundancy_idx = chunk.get('redundancy_index', -1)
            # Include redundant chunks up to redundancy_level
            # redundancy_index: 0 = first redundant, 1 = second, etc.
            # redundancy_level 0 = base only, 1 = base + first redundant, etc.
            if redundancy_idx < redundancy_level:
                gold_chunks.append(chunk)
        elif chunk_type == 'noise':
            noise_chunks.append(chunk)

    return gold_chunks, noise_chunks


def compute_qubo_energy(subset_indices, query_sim, pairwise_sim, alpha=0.05):
    """
    Compute QUBO energy for a subset of chunks.

    E = -Σᵢ relevance(i) + α·Σᵢⱼ similarity(i,j)

    Args:
        subset_indices: List of indices in the subset
        query_sim: Query similarity scores (1D array)
        pairwise_sim: Pairwise similarity matrix (2D array)
        alpha: Diversity weight

    Returns:
        energy: QUBO energy (lower is better)
    """
    # Relevance term: -Σᵢ sim(query, chunk_i)
    relevance_term = -np.sum(query_sim[subset_indices])

    # Diversity term: α·Σᵢⱼ sim(chunk_i, chunk_j) for i < j
    diversity_term = 0.0
    for i in range(len(subset_indices)):
        for j in range(i + 1, len(subset_indices)):
            idx_i = subset_indices[i]
            idx_j = subset_indices[j]
            diversity_term += pairwise_sim[idx_i, idx_j]
    diversity_term *= alpha

    energy = relevance_term + diversity_term
    return energy


def compute_gold_percentage(subset_indices, gold_indices):
    """
    Compute percentage of distinct gold facts retrieved.

    Args:
        subset_indices: List of selected chunk indices
        gold_indices: Set of gold chunk indices

    Returns:
        percentage: Percentage of gold chunks retrieved (0-100)
    """
    selected_gold = set(subset_indices) & gold_indices
    total_gold = len(gold_indices)

    if total_gold == 0:
        return 0.0

    return 100.0 * len(selected_gold) / total_gold


def enumerate_subsets(prompt_chunks, embeddings_dict, alpha=0.05, k=5, max_prompts=100):
    """
    Enumerate all K=5 subsets for prompt chunks and compute energy vs gold %.

    Args:
        prompt_chunks: List of chunks for a single prompt
        embeddings_dict: Dictionary mapping chunk_id -> embedding
        alpha: Diversity weight
        k: Number of chunks to select
        max_prompts: Maximum number of prompts to process

    Returns:
        results: Dict with 'energies', 'gold_percentages', 'prompt_results'
    """
    all_energies = []
    all_gold_percentages = []
    prompt_results = []

    # Build gold indices set
    gold_indices = set()
    for i, chunk in enumerate(prompt_chunks):
        chunk_type = chunk.get('chunk_type', '')
        if chunk_type in ['gold_base', 'gold_redundant']:
            gold_indices.add(i)

    # Get embeddings for all chunks
    chunk_embeddings = []
    for chunk in prompt_chunks:
        chunk_id = chunk['chunk_id']
        if chunk_id in embeddings_dict:
            chunk_embeddings.append(embeddings_dict[chunk_id])
        else:
            print(f"Warning: Missing embedding for chunk {chunk_id}")
            return None

    chunk_embeddings = np.array(chunk_embeddings)

    # Use first gold chunk as "query" (approximate query embedding)
    # In real scenario, we'd have actual query embedding
    if len(gold_indices) > 0:
        query_embedding = chunk_embeddings[list(gold_indices)[0]]
    else:
        query_embedding = np.mean(chunk_embeddings, axis=0)

    # Compute similarity matrices
    query_sim = np.dot(chunk_embeddings, query_embedding) / (
        np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-10
    )

    pairwise_sim = np.dot(chunk_embeddings, chunk_embeddings.T)
    pairwise_norms = np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
    pairwise_sim = pairwise_sim / (pairwise_norms @ pairwise_norms.T + 1e-10)

    # Enumerate all K-subsets
    n = len(prompt_chunks)
    total_combinations = len(list(combinations(range(n), k)))

    print(f"  Enumerating {total_combinations:,} combinations...")

    for subset in tqdm(combinations(range(n), k), total=total_combinations, desc="Subsets"):
        subset_indices = list(subset)

        # Compute QUBO energy
        energy = compute_qubo_energy(subset_indices, query_sim, pairwise_sim, alpha)

        # Compute gold percentage
        gold_pct = compute_gold_percentage(subset_indices, gold_indices)

        all_energies.append(energy)
        all_gold_percentages.append(gold_pct)

    # Normalize energies to [0, 1] for this prompt
    if len(all_energies) > 0:
        min_energy = min(all_energies)
        max_energy = max(all_energies)
        if max_energy > min_energy:
            normalized_energies = [(e - min_energy) / (max_energy - min_energy) for e in all_energies]
        else:
            normalized_energies = [0.5] * len(all_energies)
    else:
        normalized_energies = []

    return {
        'energies': all_energies,
        'normalized_energies': normalized_energies,
        'gold_percentages': all_gold_percentages,
        'n_gold': len(gold_indices),
        'n_total': n
    }


def main():
    print("=" * 80)
    print("EXPERIMENT 0.1: QUBO Energy-Quality Correlation")
    print("=" * 80)
    print()

    # Configuration
    REDUNDANCY_LEVEL = 0  # Use only gold base chunks (no redundancy)
    K = 5
    ALPHA = 0.05
    MAX_PROMPTS = 10  # Start with 10 for testing, increase to 100 for full experiment
    TARGET_POOL_SIZE = 30  # 5 gold + 25 noise

    print(f"Settings:")
    print(f"  Redundancy Level: {REDUNDANCY_LEVEL}")
    print(f"  K (chunks to select): {K}")
    print(f"  Alpha (diversity weight): {ALPHA}")
    print(f"  Max Prompts: {MAX_PROMPTS}")
    print(f"  Target Pool Size: {TARGET_POOL_SIZE}")
    print()

    # Load data
    print("Loading Wikipedia dataset...")
    chunks, embeddings_dict = load_wikipedia_data()
    print()

    # Group chunks by prompt_id
    prompts_map = defaultdict(list)
    for chunk in chunks:
        chunk_type = chunk.get('chunk_type', '')
        if chunk_type == 'prompt':
            continue  # Skip prompt chunks themselves
        prompt_id = chunk.get('prompt_id', '')
        if prompt_id:
            prompts_map[prompt_id].append(chunk)

    print(f"Found {len(prompts_map)} unique prompts")
    print()

    # Process prompts at redundancy level 0
    all_normalized_energies = []
    all_gold_percentages = []
    processed_prompts = 0

    for prompt_id, prompt_chunks_all in list(prompts_map.items())[:MAX_PROMPTS]:
        # Filter to redundancy level 0
        gold_chunks, noise_chunks = filter_chunks_for_prompt(
            prompt_chunks_all, prompt_id, redundancy_level=REDUNDANCY_LEVEL
        )

        # Ensure we have exactly 5 gold and 25 noise (or close)
        if len(gold_chunks) < 3:  # Need at least some gold chunks
            continue

        # Limit noise chunks to get target pool size
        target_noise = TARGET_POOL_SIZE - len(gold_chunks)
        if len(noise_chunks) > target_noise:
            noise_chunks = noise_chunks[:target_noise]

        prompt_chunks = gold_chunks + noise_chunks

        print(f"Prompt {processed_prompts + 1}/{MAX_PROMPTS}: {prompt_id[:30]}...")
        print(f"  Gold chunks: {len(gold_chunks)}, Noise chunks: {len(noise_chunks)}, Total: {len(prompt_chunks)}")

        if len(prompt_chunks) < K:
            print(f"  Skipping: Not enough chunks (need at least {K})")
            continue

        # Enumerate all subsets and compute energies
        results = enumerate_subsets(prompt_chunks, embeddings_dict, alpha=ALPHA, k=K)

        if results is None:
            print(f"  Skipping: Missing embeddings")
            continue

        all_normalized_energies.extend(results['normalized_energies'])
        all_gold_percentages.extend(results['gold_percentages'])

        processed_prompts += 1
        print(f"  Collected {len(results['energies']):,} data points")
        print()

    if processed_prompts == 0:
        print("ERROR: No prompts processed successfully!")
        return

    print(f"Total data points collected: {len(all_normalized_energies):,}")
    print()

    # Compute correlation
    correlation, p_value = pearsonr(all_gold_percentages, all_normalized_energies)

    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Pearson Correlation (r): {correlation:.4f}")
    print(f"P-value: {p_value:.6f}")
    print(f"Success Criterion: r < -0.7")
    print(f"Status: {'PASS' if correlation < -0.7 else 'FAIL'}")
    print()

    # Create visualization
    print("Generating plot...")

    # Group by exact gold percentage (discrete values)
    gold_percentages_array = np.array(all_gold_percentages)
    energies_array = np.array(all_normalized_energies)

    unique_percentages = np.unique(gold_percentages_array)
    bin_centers = []
    bin_means = []
    bin_stds = []

    for pct in unique_percentages:
        mask = gold_percentages_array == pct
        if np.sum(mask) > 0:
            bin_centers.append(pct)
            bin_means.append(np.mean(energies_array[mask]))
            bin_stds.append(np.std(energies_array[mask]))

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot (sample for visibility)
    sample_size = min(10000, len(all_gold_percentages))
    sample_indices = np.random.choice(len(all_gold_percentages), sample_size, replace=False)
    ax.scatter(
        [all_gold_percentages[i] for i in sample_indices],
        [all_normalized_energies[i] for i in sample_indices],
        alpha=0.1, s=1, color='blue', label='Data points'
    )

    # Error bars
    ax.errorbar(
        bin_centers, bin_means,
        yerr=bin_stds,
        fmt='o-', color='red', linewidth=2, markersize=6,
        capsize=4, capthick=2, label='Mean ± Std Dev'
    )

    # Line of best fit
    z = np.polyfit(all_gold_percentages, all_normalized_energies, 1)
    p = np.poly1d(z)
    x_fit = np.linspace(0, 100, 100)
    ax.plot(x_fit, p(x_fit), 'g--', linewidth=2, label=f'Linear Fit (r={correlation:.3f})')

    ax.set_xlabel('Gold Fact Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Normalized QUBO Energy', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Exp 0.1: QUBO Energy vs Retrieval Quality\n'
        f'(α={ALPHA}, K={K}, N≈{TARGET_POOL_SIZE}, {processed_prompts} prompts)',
        fontsize=14, fontweight='bold'
    )
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-0.05, 1.05)

    # Add text box with statistics
    textstr = f'r = {correlation:.4f}\np < {p_value:.2e}\nN = {len(all_gold_percentages):,}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    plt.tight_layout()

    # Save plot
    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / 'exp_0_1_energy_quality_correlation.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

    # Save numerical results
    results_data = {
        'correlation': float(correlation),
        'p_value': float(p_value),
        'n_data_points': int(len(all_normalized_energies)),
        'n_prompts': int(processed_prompts),
        'alpha': float(ALPHA),
        'k': int(K),
        'redundancy_level': int(REDUNDANCY_LEVEL),
        'success': bool(correlation < -0.7)
    }

    results_json_path = results_dir / 'exp_0_1_results.json'
    with open(results_json_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"Results saved to: {results_json_path}")

    print()
    print("=" * 80)
    print("EXPERIMENT 0.1 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
