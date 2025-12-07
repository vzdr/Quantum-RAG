"""
Experiment 1.2: Clean Control Test

Purpose: Prove QUBO is safe and doesn't harm performance on clean data.

Hypothesis: On datasets without redundancy (level 0), QUBO performs comparably to Top-K,
            proving it doesn't hallucinate or delete valid information.

Method:
1. Load prompts with redundancy level 0 only (no redundant chunks)
2. For each prompt:
   - Run Top-K retrieval (baseline)
   - Run QUBO retrieval (diversity-aware)
   - Measure aspect recall and gold recall
3. Compare performance between methods

Dataset: Wikipedia prompts at redundancy level 0 only
- 100 prompts
- 5 gold base chunks + 25 noise per prompt (30 total candidates)
- No redundant chunks

Settings:
- K = 5 (retrieve 5 chunks)
- QUBO parameters: α = 0.05 (diversity weight), P = 1000 (cardinality penalty)
- QUBO solver: ORBIT with 'balanced' preset

Success Metric:
- QUBO recall within ±5% of Top-K recall
- Proves QUBO doesn't introduce false negatives on clean data

Output:
- results/exp_1_2_clean_control_test.json (raw results)
- results/exp_1_2_clean_control_test.png (visualization)
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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from core.retrieval_strategies import create_retrieval_strategy


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


def filter_chunks_for_prompt(chunks, prompt_id, redundancy_level=0):
    """
    Filter chunks for a specific prompt at redundancy level 0 (clean data).

    Args:
        chunks: List of all chunks
        prompt_id: Prompt ID to filter for
        redundancy_level: Must be 0 for this experiment

    Returns:
        candidates: List of candidate chunks (gold base + noise only)
        gold_base_chunks: List of gold base chunks
        gold_base_aspects: Set of aspect_ids for gold base chunks
    """
    candidates = []
    gold_base_chunks = []
    gold_base_aspects = set()

    for chunk in chunks:
        if chunk.get('prompt_id') != prompt_id:
            continue

        chunk_type = chunk.get('chunk_type', '')

        if chunk_type == 'gold_base':
            candidates.append(chunk)
            gold_base_chunks.append(chunk)
            aspect_id = chunk.get('aspect_id', -1)
            if aspect_id >= 0:
                gold_base_aspects.add(aspect_id)
        elif chunk_type == 'noise':
            candidates.append(chunk)

    return candidates, gold_base_chunks, gold_base_aspects


def compute_metrics(selected_chunks, gold_base_chunks, gold_base_aspects):
    """
    Compute retrieval quality metrics.

    Args:
        selected_chunks: List of retrieved chunks
        gold_base_chunks: List of gold base chunks (ground truth)
        gold_base_aspects: Set of aspect_ids for the 5 distinct aspects

    Returns:
        metrics: Dictionary with aspect_recall, gold_recall, precision
    """
    # Aspect recall: how many distinct aspects were retrieved?
    retrieved_aspects = set()
    for chunk in selected_chunks:
        aspect_id = chunk.get('aspect_id', -1)
        if aspect_id >= 0:
            retrieved_aspects.add(aspect_id)

    num_gold_aspects = len(gold_base_aspects)
    num_retrieved_aspects = len(retrieved_aspects & gold_base_aspects)
    aspect_recall = 100.0 * num_retrieved_aspects / num_gold_aspects if num_gold_aspects > 0 else 0.0

    # Gold recall: how many gold chunks were retrieved?
    gold_chunk_ids = set(c['chunk_id'] for c in gold_base_chunks)
    retrieved_chunk_ids = set(c['chunk_id'] for c in selected_chunks)
    num_gold_retrieved = len(gold_chunk_ids & retrieved_chunk_ids)
    gold_recall = 100.0 * num_gold_retrieved / len(gold_chunk_ids) if len(gold_chunk_ids) > 0 else 0.0

    # Precision: what percentage of retrieved chunks are gold?
    precision = 100.0 * num_gold_retrieved / len(selected_chunks) if len(selected_chunks) > 0 else 0.0

    return {
        'aspect_recall': aspect_recall,
        'gold_recall': gold_recall,
        'precision': precision,
        'num_aspects_retrieved': num_retrieved_aspects,
        'num_gold_retrieved': num_gold_retrieved
    }


def run_clean_control_experiment(chunks, embeddings_dict, k=5, num_prompts=10):
    """
    Run Top-K vs MMR vs QUBO comparison on clean data (redundancy level 0).

    Args:
        chunks: All chunks
        embeddings_dict: All embeddings
        k: Number of chunks to retrieve
        num_prompts: Number of prompts to test (default: 10 for fast runs, use 100 for full)

    Returns:
        results: Dictionary with statistics for all methods
    """
    # Get all unique prompt IDs (extract prompt_id field from prompt chunks)
    all_prompt_ids = list(set(c['prompt_id'] for c in chunks if c.get('chunk_type') == 'prompt'))

    # Randomly sample prompts
    import random
    random.seed(42)  # Fixed seed for reproducibility
    prompt_ids = random.sample(all_prompt_ids, min(num_prompts, len(all_prompt_ids)))

    print(f"\nTesting {len(prompt_ids)} prompts (out of {len(all_prompt_ids)} total) at redundancy level 0 (clean data)")

    # Initialize strategies
    strategy_topk = create_retrieval_strategy('naive')
    strategy_mmr = create_retrieval_strategy('mmr', lambda_param=0.5)
    strategy_qubo = create_retrieval_strategy('qubo', alpha=0.05, penalty=1000.0,
                                               solver='orbit', solver_preset='balanced')

    topk_results = []
    mmr_results = []
    qubo_results = []

    for prompt_id in tqdm(prompt_ids, desc="Processing prompts", unit="prompt"):
        # Get prompt chunk (by prompt_id field, not chunk_id)
        prompt_chunks = [c for c in chunks if c.get('chunk_type') == 'prompt' and c.get('prompt_id') == prompt_id]
        if not prompt_chunks:
            continue
        prompt_chunk_id = prompt_chunks[0]['chunk_id']  # Get the actual chunk_id for embedding lookup
        prompt_embedding = embeddings_dict.get(prompt_chunk_id)
        if prompt_embedding is None:
            continue

        # Get candidates for this prompt (level 0 only: gold base + noise)
        candidates, gold_base_chunks, gold_base_aspects = filter_chunks_for_prompt(
            chunks, prompt_id, redundancy_level=0
        )

        if len(candidates) < k or len(gold_base_aspects) == 0:
            continue

        # Prepare candidate results format for retrieval strategies
        candidate_results = []
        for candidate in candidates:
            chunk_id = candidate['chunk_id']
            embedding = embeddings_dict.get(chunk_id)
            if embedding is None:
                continue

            # Compute similarity to query
            similarity = float(np.dot(prompt_embedding, embedding) /
                             (np.linalg.norm(prompt_embedding) * np.linalg.norm(embedding)))

            candidate_results.append({
                'id': chunk_id,
                'text': candidate['text'],
                'embedding': embedding,
                'score': similarity,
                'metadata': candidate
            })

        # Sort by similarity
        candidate_results.sort(key=lambda x: x['score'], reverse=True)

        # Run Top-K retrieval
        topk_retrieved, _ = strategy_topk.retrieve(
            prompt_embedding, candidate_results, k=k
        )
        topk_selected = [r.chunk.metadata for r in topk_retrieved]
        topk_metrics = compute_metrics(topk_selected, gold_base_chunks, gold_base_aspects)
        topk_results.append(topk_metrics)

        # Run MMR retrieval
        mmr_retrieved, _ = strategy_mmr.retrieve(
            prompt_embedding, candidate_results, k=k
        )
        mmr_selected = [r.chunk.metadata for r in mmr_retrieved]
        mmr_metrics = compute_metrics(mmr_selected, gold_base_chunks, gold_base_aspects)
        mmr_results.append(mmr_metrics)

        # Run QUBO retrieval
        qubo_retrieved, _ = strategy_qubo.retrieve(
            prompt_embedding, candidate_results, k=k
        )
        qubo_selected = [r.chunk.metadata for r in qubo_retrieved]
        qubo_metrics = compute_metrics(qubo_selected, gold_base_chunks, gold_base_aspects)
        qubo_results.append(qubo_metrics)

    # Aggregate statistics
    def aggregate_metrics(results_list):
        if not results_list:
            return {
                'mean_aspect_recall': 0.0,
                'std_aspect_recall': 0.0,
                'mean_gold_recall': 0.0,
                'std_gold_recall': 0.0,
                'mean_precision': 0.0,
                'std_precision': 0.0,
                'perfect_recalls': 0
            }
        return {
            'mean_aspect_recall': float(np.mean([r['aspect_recall'] for r in results_list])),
            'std_aspect_recall': float(np.std([r['aspect_recall'] for r in results_list])),
            'mean_gold_recall': float(np.mean([r['gold_recall'] for r in results_list])),
            'std_gold_recall': float(np.std([r['gold_recall'] for r in results_list])),
            'mean_precision': float(np.mean([r['precision'] for r in results_list])),
            'std_precision': float(np.std([r['precision'] for r in results_list])),
            'perfect_recalls': int(np.sum([r['aspect_recall'] == 100.0 for r in results_list]))
        }

    results = {
        'redundancy_level': 0,
        'num_prompts': len(topk_results),
        'topk': aggregate_metrics(topk_results),
        'mmr': aggregate_metrics(mmr_results),
        'qubo': aggregate_metrics(qubo_results)
    }

    print(f"\n  Top-K Aspect Recall: {results['topk']['mean_aspect_recall']:.1f}% ± {results['topk']['std_aspect_recall']:.1f}%")
    print(f"  MMR Aspect Recall:   {results['mmr']['mean_aspect_recall']:.1f}% ± {results['mmr']['std_aspect_recall']:.1f}%")
    print(f"  QUBO Aspect Recall:  {results['qubo']['mean_aspect_recall']:.1f}% ± {results['qubo']['std_aspect_recall']:.1f}%")

    return results


def plot_results(results, output_path):
    """
    Generate visualization comparing Top-K vs MMR vs QUBO on clean data.

    Args:
        results: Result dictionary
        output_path: Path to save the plot
    """
    metrics = ['Aspect Recall', 'Gold Recall', 'Precision']
    topk_values = [
        results['topk']['mean_aspect_recall'],
        results['topk']['mean_gold_recall'],
        results['topk']['mean_precision']
    ]
    topk_errors = [
        results['topk']['std_aspect_recall'],
        results['topk']['std_gold_recall'],
        results['topk']['std_precision']
    ]
    mmr_values = [
        results['mmr']['mean_aspect_recall'],
        results['mmr']['mean_gold_recall'],
        results['mmr']['mean_precision']
    ]
    mmr_errors = [
        results['mmr']['std_aspect_recall'],
        results['mmr']['std_gold_recall'],
        results['mmr']['std_precision']
    ]
    qubo_values = [
        results['qubo']['mean_aspect_recall'],
        results['qubo']['mean_gold_recall'],
        results['qubo']['mean_precision']
    ]
    qubo_errors = [
        results['qubo']['std_aspect_recall'],
        results['qubo']['std_gold_recall'],
        results['qubo']['std_precision']
    ]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(metrics))
    width = 0.25

    # Plot bars with error bars
    ax.bar(x - width, topk_values, width, yerr=topk_errors,
           label='Top-K (Baseline)', alpha=0.8, capsize=5, color='#e74c3c')
    ax.bar(x, mmr_values, width, yerr=mmr_errors,
           label='MMR (SOTA)', alpha=0.8, capsize=5, color='#f39c12')
    ax.bar(x + width, qubo_values, width, yerr=qubo_errors,
           label='QUBO (Ours)', alpha=0.8, capsize=5, color='#3498db')

    # Formatting
    ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Experiment 1.2: Clean Control Test\nTop-K vs MMR vs QUBO on Clean Data',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(loc='best', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")


def main():
    """Run Experiment 1.2: Clean Control Test."""
    import argparse
    parser = argparse.ArgumentParser(description='Experiment 1.2: Clean Control Test')
    parser.add_argument('--num-prompts', type=int, default=10,
                        help='Number of prompts to test (default: 10, use 100 for full experiment)')
    args = parser.parse_args()

    print("="*80)
    print("EXPERIMENT 1.2: CLEAN CONTROL TEST")
    print("="*80)
    print(f"Testing with {args.num_prompts} prompts (use --num-prompts 100 for full run)")
    print("="*80)

    # Load data
    chunks, embeddings_dict = load_wikipedia_data()

    # Run experiment on clean data (redundancy level 0)
    results = run_clean_control_experiment(chunks, embeddings_dict, k=5, num_prompts=args.num_prompts)

    # Save results
    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)

    output_json = results_dir / 'exp_1_2_clean_control_test.json'
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_json}")

    # Generate plot
    output_plot = results_dir / 'exp_1_2_clean_control_test.png'
    plot_results(results, output_plot)

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Top-K Aspect Recall: {results['topk']['mean_aspect_recall']:.1f}% ± {results['topk']['std_aspect_recall']:.1f}%")
    print(f"MMR Aspect Recall:   {results['mmr']['mean_aspect_recall']:.1f}% ± {results['mmr']['std_aspect_recall']:.1f}%")
    print(f"QUBO Aspect Recall:  {results['qubo']['mean_aspect_recall']:.1f}% ± {results['qubo']['std_aspect_recall']:.1f}%")

    # Success criteria check
    print("\n" + "="*80)
    print("SUCCESS CRITERIA")
    print("="*80)

    difference = abs(results['qubo']['mean_aspect_recall'] - results['topk']['mean_aspect_recall'])
    success = difference <= 5.0

    print(f"✓ QUBO within ±5% of Top-K: {difference:.1f}% - {'PASS' if success else 'FAIL'}")
    print(f"\nOverall: {'✓ SUCCESS' if success else '✗ FAIL'}")

    if success:
        print("\nConclusion: QUBO is safe and does not harm performance on clean data.")
    else:
        print(f"\nWarning: QUBO differs by {difference:.1f}% from Top-K on clean data.")


if __name__ == '__main__':
    main()
