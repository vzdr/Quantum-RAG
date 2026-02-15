"""
Experiment 0.2: Redundancy Robustness

Purpose: Validate QUBO energy correlation holds across all redundancy levels 0-5.

Hypothesis: QUBO energy consistently predicts distinct gold fact retrieval regardless
of redundancy level, proving parameters aren't overfit to the no-redundancy case.

Method:
1. For each redundancy level (0-5):
   - Sample random prompts
   - Filter chunks to include gold (base + redundant up to level N) + noise
   - Enumerate combinations, compute energies
   - Normalize energies per prompt per level
   - Plot distinct facts vs energy for that level

Dataset: Random subset of Wikipedia prompts (10-20 per level for testing, 100 for full)

Settings:
- Redundancy levels: 0-5
- K = 5 chunks selected
- Candidate pool varies by level (e.g., level 0: 30 chunks, level 5: ~55 chunks)
- Alpha = 0.05 (diversity weight)

Success Metric: Correlation r < -0.7 for all redundancy levels
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


def load_wikipedia_data():
    """Load Wikipedia dataset (chunks, embeddings)."""
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

    return chunks, embeddings_dict


def filter_chunks_for_prompt(chunks, prompt_id, redundancy_level):
    """
    Filter chunks for a specific prompt at a given redundancy level.

    Args:
        chunks: List of all chunks
        prompt_id: Prompt ID to filter for
        redundancy_level: Maximum redundancy level to include
                         0 = base only
                         1 = base + 1 redundant per aspect
                         5 = base + 5 redundant per aspect

    Returns:
        gold_chunks: List of gold chunks (base + redundant up to level)
        noise_chunks: List of noise chunks
        n_distinct_aspects: Number of distinct aspects (base chunks)
    """
    gold_chunks = []
    noise_chunks = []
    distinct_aspects = set()

    for chunk in chunks:
        if chunk.get('prompt_id') != prompt_id:
            continue

        chunk_type = chunk.get('chunk_type', '')
        aspect_id = chunk.get('aspect_id', -1)

        if chunk_type == 'gold_base':
            gold_chunks.append(chunk)
            distinct_aspects.add(aspect_id)
        elif chunk_type == 'gold_redundant':
            redundancy_idx = chunk.get('redundancy_index', -1)
            # redundancy_index starts at 0 for first redundant
            # Include if redundancy_idx < redundancy_level
            if redundancy_idx < redundancy_level:
                gold_chunks.append(chunk)
        elif chunk_type == 'noise':
            noise_chunks.append(chunk)

    return gold_chunks, noise_chunks, len(distinct_aspects)


def compute_qubo_energy(subset_indices, query_sim, pairwise_sim, alpha=0.05):
    """
    Compute QUBO energy for a subset.

    E = -Σᵢ relevance(i) + α·Σᵢⱼ similarity(i,j)
    """
    # Relevance term
    relevance_term = -np.sum(query_sim[subset_indices])

    # Diversity term
    diversity_term = 0.0
    for i in range(len(subset_indices)):
        for j in range(i + 1, len(subset_indices)):
            idx_i = subset_indices[i]
            idx_j = subset_indices[j]
            diversity_term += pairwise_sim[idx_i, idx_j]
    diversity_term *= alpha

    energy = relevance_term + diversity_term
    return energy


def compute_distinct_facts(subset_indices, chunk_aspect_ids):
    """
    Compute number of distinct aspects (facts) retrieved.

    Args:
        subset_indices: List of selected chunk indices
        chunk_aspect_ids: List mapping chunk index -> aspect_id (-1 for noise)

    Returns:
        n_distinct: Number of distinct aspects in subset
    """
    aspects = set()
    for idx in subset_indices:
        aspect_id = chunk_aspect_ids[idx]
        if aspect_id >= 0:  # Gold chunk
            aspects.add(aspect_id)

    return len(aspects)


def enumerate_subsets_for_level(prompt_chunks, embeddings_dict, alpha=0.05, k=5,
                                 max_combinations=50000):
    """
    Enumerate subsets for a prompt at a specific redundancy level.

    Args:
        prompt_chunks: Chunks for this prompt at this redundancy level
        embeddings_dict: Embedding dictionary
        alpha: Diversity weight
        k: Number of chunks to select
        max_combinations: Maximum combinations to sample (for computational tractability)

    Returns:
        results: Dict with energies, normalized energies, distinct facts
    """
    # Extract aspect IDs for each chunk
    chunk_aspect_ids = []
    for chunk in prompt_chunks:
        chunk_type = chunk.get('chunk_type', '')
        if chunk_type in ['gold_base', 'gold_redundant']:
            chunk_aspect_ids.append(chunk.get('aspect_id', -1))
        else:
            chunk_aspect_ids.append(-1)  # Noise

    # Get embeddings
    chunk_embeddings = []
    for chunk in prompt_chunks:
        chunk_id = chunk['chunk_id']
        if chunk_id in embeddings_dict:
            chunk_embeddings.append(embeddings_dict[chunk_id])
        else:
            return None

    chunk_embeddings = np.array(chunk_embeddings)

    # Use mean of gold chunks as query approximation
    gold_indices = [i for i, aid in enumerate(chunk_aspect_ids) if aid >= 0]
    if len(gold_indices) > 0:
        query_embedding = np.mean(chunk_embeddings[gold_indices], axis=0)
    else:
        query_embedding = np.mean(chunk_embeddings, axis=0)

    # Compute similarities
    query_sim = np.dot(chunk_embeddings, query_embedding) / (
        np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-10
    )

    pairwise_sim = np.dot(chunk_embeddings, chunk_embeddings.T)
    pairwise_norms = np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
    pairwise_sim = pairwise_sim / (pairwise_norms @ pairwise_norms.T + 1e-10)

    # Generate all combinations or sample
    n = len(prompt_chunks)
    all_combinations = list(combinations(range(n), k))
    total_combinations = len(all_combinations)

    if total_combinations > max_combinations:
        # Sample randomly
        import random
        sampled_combinations = random.sample(all_combinations, max_combinations)
        print(f"    Sampling {max_combinations:,} of {total_combinations:,} combinations")
    else:
        sampled_combinations = all_combinations
        print(f"    Enumerating {total_combinations:,} combinations")

    energies = []
    distinct_facts = []

    for subset in tqdm(sampled_combinations, desc="    Subsets", leave=False):
        subset_indices = list(subset)

        # Compute energy
        energy = compute_qubo_energy(subset_indices, query_sim, pairwise_sim, alpha)

        # Compute distinct facts
        n_distinct = compute_distinct_facts(subset_indices, chunk_aspect_ids)

        energies.append(energy)
        distinct_facts.append(n_distinct)

    # Normalize energies to [0, 1]
    if len(energies) > 0:
        min_energy = min(energies)
        max_energy = max(energies)
        if max_energy > min_energy:
            normalized_energies = [(e - min_energy) / (max_energy - min_energy) for e in energies]
        else:
            normalized_energies = [0.5] * len(energies)
    else:
        normalized_energies = []

    return {
        'energies': energies,
        'normalized_energies': normalized_energies,
        'distinct_facts': distinct_facts,
        'n_chunks': n,
        'n_combinations': len(energies)
    }


def process_redundancy_level(chunks, embeddings_dict, redundancy_level,
                             max_prompts=10, alpha=0.05, k=5):
    """
    Process all prompts at a specific redundancy level.

    Returns:
        all_normalized_energies, all_distinct_facts, n_prompts
    """
    # Group chunks by prompt
    prompts_map = defaultdict(list)
    for chunk in chunks:
        chunk_type = chunk.get('chunk_type', '')
        if chunk_type == 'prompt':
            continue
        prompt_id = chunk.get('prompt_id', '')
        if prompt_id:
            prompts_map[prompt_id].append(chunk)

    all_normalized_energies = []
    all_distinct_facts = []
    processed_prompts = 0

    for prompt_id in list(prompts_map.keys())[:max_prompts]:
        prompt_chunks_all = prompts_map[prompt_id]

        # Filter to redundancy level
        gold_chunks, noise_chunks, n_aspects = filter_chunks_for_prompt(
            prompt_chunks_all, prompt_id, redundancy_level
        )

        if n_aspects < 3:  # Need at least some aspects
            continue

        # Limit noise to ~25 chunks
        if len(noise_chunks) > 25:
            noise_chunks = noise_chunks[:25]

        prompt_chunks = gold_chunks + noise_chunks

        if len(prompt_chunks) < k:
            continue

        print(f"  Prompt {processed_prompts + 1}: {len(gold_chunks)} gold, "
              f"{len(noise_chunks)} noise, {n_aspects} aspects")

        # Enumerate subsets
        results = enumerate_subsets_for_level(
            prompt_chunks, embeddings_dict, alpha=alpha, k=k
        )

        if results is None:
            continue

        all_normalized_energies.extend(results['normalized_energies'])
        all_distinct_facts.extend(results['distinct_facts'])

        processed_prompts += 1

        if processed_prompts >= max_prompts:
            break

    return all_normalized_energies, all_distinct_facts, processed_prompts


def main():
    print("=" * 80)
    print("EXPERIMENT 0.2: Redundancy Robustness")
    print("=" * 80)
    print()

    # Configuration
    REDUNDANCY_LEVELS = [0, 1, 2, 3, 4, 5]
    K = 5
    ALPHA = 0.05
    MAX_PROMPTS_PER_LEVEL = 5  # Start with 5 for testing, increase to 20-100 for full

    print(f"Settings:")
    print(f"  Redundancy Levels: {REDUNDANCY_LEVELS}")
    print(f"  K (chunks to select): {K}")
    print(f"  Alpha (diversity weight): {ALPHA}")
    print(f"  Max Prompts per Level: {MAX_PROMPTS_PER_LEVEL}")
    print()

    # Load data
    print("Loading Wikipedia dataset...")
    chunks, embeddings_dict = load_wikipedia_data()
    print()

    # Results storage
    level_results = {}

    # Process each redundancy level
    for level in REDUNDANCY_LEVELS:
        print(f"{'=' * 80}")
        print(f"Processing Redundancy Level {level}")
        print(f"{'=' * 80}")

        energies, distinct_facts, n_prompts = process_redundancy_level(
            chunks, embeddings_dict, level,
            max_prompts=MAX_PROMPTS_PER_LEVEL, alpha=ALPHA, k=K
        )

        if len(energies) == 0:
            print(f"WARNING: No data collected for level {level}")
            continue

        # Compute correlation
        correlation, p_value = pearsonr(distinct_facts, energies)

        level_results[level] = {
            'energies': energies,
            'distinct_facts': distinct_facts,
            'correlation': correlation,
            'p_value': p_value,
            'n_prompts': n_prompts,
            'n_data_points': len(energies)
        }

        print(f"Level {level} Results:")
        print(f"  Data points: {len(energies):,}")
        print(f"  Prompts processed: {n_prompts}")
        print(f"  Correlation (r): {correlation:.4f}")
        print(f"  P-value: {p_value:.6f}")
        print(f"  Status: {'PASS' if correlation < -0.7 else 'FAIL'}")
        print()

    # Generate visualizations
    print("=" * 80)
    print("Generating Plots...")
    print("=" * 80)

    # Create subplot grid (2x3 for 6 levels)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, level in enumerate(REDUNDANCY_LEVELS):
        if level not in level_results:
            continue

        ax = axes[idx]
        results = level_results[level]

        energies = results['energies']
        distinct_facts = results['distinct_facts']
        correlation = results['correlation']
        p_value = results['p_value']

        # Scatter plot (sample for visibility)
        sample_size = min(5000, len(energies))
        sample_indices = np.random.choice(len(energies), sample_size, replace=False)
        ax.scatter(
            [distinct_facts[i] for i in sample_indices],
            [energies[i] for i in sample_indices],
            alpha=0.2, s=2, color='blue'
        )

        # Group by exact distinct fact count (discrete values)
        distinct_facts_array = np.array(distinct_facts)
        energies_array = np.array(energies)

        unique_facts = np.unique(distinct_facts_array)
        bin_centers = []
        bin_means = []
        bin_stds = []

        for fact_count in unique_facts:
            mask = distinct_facts_array == fact_count
            if np.sum(mask) > 0:
                bin_centers.append(fact_count)
                bin_means.append(np.mean(energies_array[mask]))
                bin_stds.append(np.std(energies_array[mask]))

        if len(bin_means) > 0:
            ax.errorbar(bin_centers, bin_means, yerr=bin_stds,
                       fmt='o-', color='red', linewidth=2, markersize=6,
                       capsize=4, capthick=2, label='Mean ± Std')

        # Line of best fit
        if len(distinct_facts) > 1:
            z = np.polyfit(distinct_facts, energies, 1)
            p = np.poly1d(z)
            x_fit = np.linspace(min(distinct_facts), max(distinct_facts), 50)
            ax.plot(x_fit, p(x_fit), 'g--', linewidth=2, label=f'Fit (r={correlation:.3f})')

        ax.set_xlabel('Distinct Facts Retrieved', fontsize=10, fontweight='bold')
        ax.set_ylabel('Normalized QUBO Energy', fontsize=10, fontweight='bold')
        ax.set_title(f'Level {level}\n(r={correlation:.3f}, p<{p_value:.2e})',
                    fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

        # Color-code title based on success
        if correlation < -0.7:
            ax.title.set_color('green')
        else:
            ax.title.set_color('red')

    plt.suptitle(f'Exp 0.2: QUBO Energy vs Distinct Facts Across Redundancy Levels\n'
                f'(α={ALPHA}, K={K})', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save plot
    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / 'exp_0_2_redundancy_robustness.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

    # Save numerical results
    results_summary = {
        'alpha': float(ALPHA),
        'k': int(K),
        'max_prompts_per_level': int(MAX_PROMPTS_PER_LEVEL),
        'levels': {}
    }

    for level, results in level_results.items():
        results_summary['levels'][str(level)] = {
            'correlation': float(results['correlation']),
            'p_value': float(results['p_value']),
            'n_data_points': int(results['n_data_points']),
            'n_prompts': int(results['n_prompts']),
            'success': bool(results['correlation'] < -0.7)
        }

    results_json_path = results_dir / 'exp_0_2_results.json'
    with open(results_json_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"Results saved to: {results_json_path}")

    # Print summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    success_count = sum(1 for r in level_results.values() if r['correlation'] < -0.7)
    print(f"Levels passing criterion (r < -0.7): {success_count}/{len(level_results)}")
    print()
    for level in REDUNDANCY_LEVELS:
        if level in level_results:
            r = level_results[level]
            status = 'PASS' if r['correlation'] < -0.7 else 'FAIL'
            print(f"  Level {level}: r = {r['correlation']:.4f} {status}")
    print()
    print("=" * 80)
    print("EXPERIMENT 0.2 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
