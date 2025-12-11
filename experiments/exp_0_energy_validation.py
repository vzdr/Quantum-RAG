"""
Experiment 0: Redundancy Robustness

Purpose: Validate QUBO energy correlation holds across all redundancy levels (0-5).
Hypothesis: QUBO energy consistently predicts distinct gold fact retrieval regardless
of redundancy level.
"""
import sys
import json
import numpy as np
from pathlib import Path
from itertools import combinations
from tqdm import tqdm
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

# Add project root to path and import core modules
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from core.utils import load_wikipedia_dataset, filter_chunks_by_prompt

def compute_qubo_energy(subset_indices, query_sim, pairwise_sim, alpha):
    """Computes the QUBO energy for a given subset of chunks."""
    relevance_term = -np.sum(query_sim[subset_indices])
    
    diversity_term = 0.0
    for i in range(len(subset_indices)):
        for j in range(i + 1, len(subset_indices)):
            diversity_term += pairwise_sim[subset_indices[i], subset_indices[j]]
    
    return relevance_term + (alpha * diversity_term)

def run_prompt_enumeration(all_prompt_chunks, embeddings_dict, alpha, k):
    """Enumerates all subsets for a single prompt's chunks and returns energy/quality pairs."""
    chunk_embeddings = np.array([embeddings_dict[c['chunk_id']] for c in all_prompt_chunks])
    
    # Use the mean of gold chunks as an approximate query embedding
    gold_indices = [i for i, c in enumerate(all_prompt_chunks) if c.get('chunk_type') in ['gold_base', 'gold_redundant']]
    query_embedding = np.mean(chunk_embeddings[gold_indices], axis=0) if gold_indices else np.mean(chunk_embeddings, axis=0)
    
    # Normalize and compute similarities
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    cand_norms = chunk_embeddings / np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
    query_sim = np.dot(cand_norms, query_norm)
    pairwise_sim = np.dot(cand_norms, cand_norms.T)

    energies, distinct_facts_list = [], []
    for subset in combinations(range(len(all_prompt_chunks)), k):
        subset_indices = list(subset)
        energy = compute_qubo_energy(subset_indices, query_sim, pairwise_sim, alpha)
        
        selected_meta = [all_prompt_chunks[i] for i in subset_indices]
        # Calculate distinct facts using compute_aspect_recall's internal logic
        retrieved_aspects = set()
        for chunk_meta in selected_meta:
            aspect_id = chunk_meta.get('aspect_id', -1)
            if aspect_id >= 0:
                retrieved_aspects.add(aspect_id)
        distinct_facts_list.append(len(retrieved_aspects))

        energies.append(energy)
        
    # Normalize energies for this prompt
    min_e, max_e = min(energies), max(energies)
    normalized_energies = [(e - min_e) / (max_e - min_e) if max_e > min_e else 0.5 for e in energies]
    
    return normalized_energies, distinct_facts_list

def plot_results(level_results, config, output_path):
    """Generates and saves the correlation plots for each redundancy level."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, level in enumerate(config['redundancy_levels']):
        ax = axes[idx]
        if level not in level_results:
            ax.set_title(f'Level {level} (No Data)', color='red'); continue
        
        results = level_results[level]
        energies_arr = np.array(results['energies'])
        distinct_facts_arr = np.array(results['distinct_facts'])
        correlation = results['correlation']
        p_value = results['p_value']

        # Scatter plot (sample for visibility)
        sample_size = min(5000, len(energies_arr))
        sample_indices = np.random.choice(len(energies_arr), sample_size, replace=False)
        ax.scatter(distinct_facts_arr[sample_indices], energies_arr[sample_indices], alpha=0.2, s=2, color='blue')

        # Mean and error bars
        unique_facts = np.unique(distinct_facts_arr)
        bin_means = [np.mean(energies_arr[distinct_facts_arr == f]) for f in unique_facts]
        bin_stds = [np.std(energies_arr[distinct_facts_arr == f]) for f in unique_facts]
        if unique_facts.size > 0:
            ax.errorbar(unique_facts, bin_means, yerr=bin_stds, fmt='o-', color='red', label='Mean ± Std')

        # Line of best fit
        if distinct_facts_arr.size > 1:
            z = np.polyfit(distinct_facts_arr, energies_arr, 1)
            p = np.poly1d(z)
            ax.plot(unique_facts, p(unique_facts), 'g--', label=f'Fit (r={correlation:.3f})')

        ax.set_xlabel('Distinct Facts Retrieved')
        ax.set_ylabel('Normalized QUBO Energy')
        ax.set_title(f'Level {level}\n(r={correlation:.3f}, p<{p_value:.2e})', color='green' if correlation < -0.7 else 'red')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'QUBO Energy vs Distinct Facts Across Redundancy Levels (α={config["alpha"]}, K={config["k"]})', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Experiment 0.2: Redundancy Robustness')
    parser.add_argument('--num-prompts-per-level', type=int, default=5, help='Number of prompts to process per level.')
    parser.add_argument('--k', type=int, default=5, help='Number of chunks to select.')
    parser.add_argument('--alpha', type=float, default=0.1, help='Diversity weight.')
    parser.add_argument('--redundancy-levels', type=int, nargs='+', default=list(range(6)), help='Redundancy levels to test.')
    args = parser.parse_args()

    print("--- Running Experiment 0.2: Redundancy Robustness ---")
    chunks, embeddings_dict = load_wikipedia_dataset()

    level_results = {}
    all_prompt_ids = list(set(c['prompt_id'] for c in chunks if c.get('chunk_type') == 'prompt'))
    
    for level in args.redundancy_levels:
        print(f"\n--- Processing Redundancy Level {level} ---")
        
        # Sample prompts for this level
        np.random.seed(42) # Ensure reproducibility for sampled prompts per level
        sampled_prompt_ids = np.random.choice(all_prompt_ids, size=min(args.num_prompts_per_level, len(all_prompt_ids)), replace=False)
        
        level_energies, level_distinct_facts = [], []
        
        for prompt_id in tqdm(sampled_prompt_ids, desc=f"Level {level} Prompts"):
            _, gold_aspects, gold_chunks, noise_chunks, n_distinct_aspects = filter_chunks_by_prompt(
                chunks, prompt_id, redundancy_level=level
            )
            
            # Create a pool of chunks for enumeration: gold + a sample of noise
            if n_distinct_aspects < 3: continue # Ensure enough distinct aspects
            
            # Limit noise to keep candidate pool manageable for brute-force (around 30-50 total)
            target_noise_count = max(0, 30 - len(gold_chunks)) # Adjust based on gold chunks
            if len(noise_chunks) > target_noise_count:
                noise_chunks = list(np.random.choice(noise_chunks, target_noise_count, replace=False))
            
            prompt_chunks_for_enum = gold_chunks + noise_chunks
            
            if len(prompt_chunks_for_enum) < args.k: continue

            energies, distinct_facts = run_prompt_enumeration(
                prompt_chunks_for_enum, embeddings_dict, args.alpha, args.k
            )
            level_energies.extend(energies)
            level_distinct_facts.extend(distinct_facts)
        
        if not level_energies:
            print(f"  No data collected for level {level}. Skipping.")
            continue
            
        correlation, p_value = pearsonr(level_distinct_facts, level_energies)
        level_results[level] = {
            'energies': level_energies,
            'distinct_facts': level_distinct_facts,
            'correlation': correlation,
            'p_value': p_value,
            'n_prompts': len(sampled_prompt_ids),
            'n_data_points': len(level_energies)
        }
        print(f"  Correlation (r): {correlation:.4f}, p-value: {p_value:.2e}. Status: {'PASS' if correlation < -0.7 else 'FAIL'}")

    # Save results and plot
    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / 'exp_0_results.json', 'w') as f:
        json.dump({
            'alpha': args.alpha, 'k': args.k, 'num_prompts_per_level': args.num_prompts_per_level,
            'levels': {str(level): {k:v for k,v in res.items() if k not in ['energies', 'distinct_facts']} 
                       for level, res in level_results.items()}
        }, f, indent=2)

    plot_results(level_results, vars(args), results_dir / 'exp_0_energy_validation.png')

    print("\n--- Summary ---")
    success_count = sum(1 for r in level_results.values() if r['correlation'] < -0.7)
    print(f"Levels passing criterion (r < -0.7): {success_count}/{len(level_results)}")
    for level, res in level_results.items():
        print(f"  Level {level}: r = {res['correlation']:.4f} ({'PASS' if res['correlation'] < -0.7 else 'FAIL'})")
    
    print("\n--- Experiment Complete ---")

if __name__ == "__main__":
    main()
