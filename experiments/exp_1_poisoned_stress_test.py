"""
Experiment 1: Poisoned Stress Test

Purpose: Demonstrate that QUBO retrieval maintains aspect recall under redundancy
         while Top-K and MMR degrade.
"""
import sys
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

# Add project root to path and import core modules
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from core.retrieval import NaiveRetrieval, MMRRetrieval, QUBORetrieval
from core.utils import (
    load_wikipedia_dataset,
    filter_chunks_by_prompt,
    get_prompt_embedding,
    compute_aspect_recall
)

def run_retrieval_comparison(chunks, embeddings_dict, redundancy_level, k=5, num_prompts=10):
    """Runs Top-K, MMR, and QUBO comparison for a given redundancy level."""
    all_prompt_ids = list(set(c['prompt_id'] for c in chunks if c.get('chunk_type') == 'prompt'))
    prompt_ids = np.random.choice(all_prompt_ids, size=min(num_prompts, len(all_prompt_ids)), replace=False).tolist()

    print(f"\nRunning Level {redundancy_level} with {len(prompt_ids)} prompts...")

    strategies = {
        'topk': NaiveRetrieval(),
        'mmr': MMRRetrieval(lambda_param=0.85),
        'qubo': QUBORetrieval(alpha=0.04, penalty=1, solver='gurobi')
    }
    
    recalls = {name: [] for name in strategies}

    for prompt_id in tqdm(prompt_ids, desc=f"Level {redundancy_level}", unit="prompt", leave=False):
        prompt_embedding = get_prompt_embedding(chunks, embeddings_dict, prompt_id)
        if prompt_embedding is None:
            continue

        candidates, gold_aspects, _, _, _ = filter_chunks_by_prompt(chunks, prompt_id, redundancy_level)
        if len(candidates) < k or not gold_aspects:
            continue

        candidate_results = [{
            'id': cand['chunk_id'],
            'text': cand['text'],
            'embedding': embeddings_dict.get(cand['chunk_id']),
            'score': np.dot(prompt_embedding / np.linalg.norm(prompt_embedding),
                            embeddings_dict.get(cand['chunk_id']) / np.linalg.norm(embeddings_dict.get(cand['chunk_id']))),
            'metadata': cand
        } for cand in candidates if embeddings_dict.get(cand['chunk_id']) is not None]
        
        candidate_results.sort(key=lambda x: x['score'], reverse=True)

        for name, strategy in strategies.items():
            selected = strategy.retrieve(prompt_embedding, candidate_results, k=k)
            selected_meta = [r.chunk.metadata for r in selected]
            recall, _ = compute_aspect_recall(selected_meta, gold_aspects)
            recalls[name].append(recall)

    # Aggregate and return statistics
    agg_results = {}
    for name, recall_list in recalls.items():
        agg_results[name] = {
            'mean_aspect_recall': np.mean(recall_list) if recall_list else 0.0,
            'std_aspect_recall': np.std(recall_list) if recall_list else 0.0,
        }
        print(f"  {name.upper()}: {agg_results[name]['mean_aspect_recall']:.1f}% ± {agg_results[name]['std_aspect_recall']:.1f}%")
        
    return {'redundancy_level': redundancy_level, 'num_prompts': len(prompt_ids), **agg_results}

def plot_results(all_results, output_path):
    """Generates and saves a comparison plot."""
    redundancy_levels = [r['redundancy_level'] for r in all_results]
    topk_means = [r['topk']['mean_aspect_recall'] for r in all_results]
    topk_stds = [r['topk']['std_aspect_recall'] for r in all_results]
    mmr_means = [r['mmr']['mean_aspect_recall'] for r in all_results]
    mmr_stds = [r['mmr']['std_aspect_recall'] for r in all_results]
    qubo_means = [r['qubo']['mean_aspect_recall'] for r in all_results]
    qubo_stds = [r['qubo']['std_aspect_recall'] for r in all_results]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.array(redundancy_levels)
    width = 0.25 # Adjust width for 3 bars per group

    # Plot bars with error bars
    ax.bar(x - width, topk_means, width, yerr=topk_stds,
           label='Top-K (Baseline)', alpha=0.8, capsize=5, color='#e74c3c')
    ax.bar(x, mmr_means, width, yerr=mmr_stds,
           label='MMR (SOTA)', alpha=0.8, capsize=5, color='#f39c12')
    ax.bar(x + width, qubo_means, width, yerr=qubo_stds,
           label='QUBO (Ours)', alpha=0.8, capsize=5, color='#3498db')

    # Formatting
    ax.set_xlabel('Redundancy Level', fontsize=12, fontweight='bold')
    ax.set_ylabel('Aspect Recall (%)', fontsize=12, fontweight='bold')
    ax.set_title('Experiment 1: Poisoned Stress Test\nTop-K vs MMR vs QUBO',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{i}' for i in redundancy_levels])
    ax.legend(loc='best', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"\nPlot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Experiment 1: Poisoned Stress Test')
    parser.add_argument('--num-prompts', type=int, default=10, help='Number of prompts to test.')
    args = parser.parse_args()

    print("--- Running Experiment 1: Poisoned Stress Test ---")
    chunks, embeddings_dict = load_wikipedia_dataset()

    all_results = [
        run_retrieval_comparison(chunks, embeddings_dict, level, k=5, num_prompts=args.num_prompts)
        for level in range(6)
    ]

    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)
    output_json = results_dir / 'exp_1_poisoned_stress_test.json'
    with open(output_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_json}")

    plot_results(all_results, results_dir / 'exp_1_poisoned_stress_test.png')

    print("\n--- Summary ---")
    for result in all_results:
        print(f"Level {result['redundancy_level']}: "
              f"Top-K={result['topk']['mean_aspect_recall']:.1f}%, "
              f"MMR={result['mmr']['mean_aspect_recall']:.1f}%, "
              f"QUBO={result['qubo']['mean_aspect_recall']:.1f}%")
    print("--- Experiment Complete ---")

if __name__ == '__main__':
    main()
