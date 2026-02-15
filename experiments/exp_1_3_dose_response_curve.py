"""
Experiment 1.3: Dose-Response Curve

Purpose: Show QUBO advantage scales continuously with redundancy level.

Hypothesis: Even mild redundancy degrades Top-K performance, while QUBO remains
            stable across all redundancy levels.

Method:
1. For each redundancy level (0-5):
   - Run Top-K and QUBO retrieval
   - Measure aspect recall for both
   - Plot continuous curve showing degradation vs stability
2. Analyze the "dose-response" relationship between redundancy and performance

Dataset: Same 100 prompts tested at each redundancy level
- Level 0: 5 gold base (no redundancy) + 25 noise
- Level 1: 5 gold base + 5 redundant (1 per aspect) + 25 noise
- Level 2: 5 gold base + 10 redundant (2 per aspect) + 25 noise
- Level 3: 5 gold base + 15 redundant (3 per aspect) + 25 noise
- Level 5: 5 gold base + 25 redundant (5 per aspect) + 25 noise

Settings:
- K = 5 (retrieve 5 chunks)
- QUBO parameters: α = 0.05 (diversity weight), P = 1000 (cardinality penalty)
- QUBO solver: ORBIT with 'balanced' preset

Success Metrics:
- Top-K recall drops >20% with just redundancy level 1
- QUBO recall remains flat within ±5% across all levels

Output:
- results/exp_1_3_dose_response_curve.json (raw results)
- results/exp_1_3_dose_response_curve.png (line plot showing curves)
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
    Filter chunks for a specific prompt at a given redundancy level.

    Args:
        chunks: List of all chunks
        prompt_id: Prompt ID to filter for
        redundancy_level: Maximum redundancy level to include

    Returns:
        candidates: List of candidate chunks
        gold_base_aspects: Set of aspect_ids for gold base chunks
    """
    candidates = []
    gold_base_aspects = set()

    for chunk in chunks:
        if chunk.get('prompt_id') != prompt_id:
            continue

        chunk_type = chunk.get('chunk_type', '')

        if chunk_type == 'gold_base':
            candidates.append(chunk)
            aspect_id = chunk.get('aspect_id', -1)
            if aspect_id >= 0:
                gold_base_aspects.add(aspect_id)
        elif chunk_type == 'gold_redundant':
            redundancy_idx = chunk.get('redundancy_index', -1)
            if redundancy_idx < redundancy_level:
                candidates.append(chunk)
        elif chunk_type == 'noise':
            candidates.append(chunk)

    return candidates, gold_base_aspects


def compute_aspect_recall(selected_chunks, gold_base_aspects):
    """
    Compute aspect recall: how many distinct aspects were retrieved?

    Args:
        selected_chunks: List of retrieved chunks
        gold_base_aspects: Set of aspect_ids for the 5 distinct aspects

    Returns:
        aspect_recall: Percentage of distinct aspects retrieved (0-100)
    """
    retrieved_aspects = set()
    for chunk in selected_chunks:
        aspect_id = chunk.get('aspect_id', -1)
        if aspect_id >= 0:
            retrieved_aspects.add(aspect_id)

    num_gold_aspects = len(gold_base_aspects)
    num_retrieved = len(retrieved_aspects & gold_base_aspects)

    if num_gold_aspects == 0:
        return 0.0

    return 100.0 * num_retrieved / num_gold_aspects


def run_dose_response_experiment(chunks, embeddings_dict, redundancy_levels, k=5, num_prompts=10):
    """
    Run dose-response analysis across all redundancy levels.

    Args:
        chunks: All chunks
        embeddings_dict: All embeddings
        redundancy_levels: List of redundancy levels to test
        k: Number of chunks to retrieve
        num_prompts: Number of prompts to test (default: 10 for fast runs, use 100 for full)

    Returns:
        all_results: List of result dictionaries for each level
    """
    # Get all unique prompt IDs (extract prompt_id field from prompt chunks)
    all_prompt_ids = list(set(c['prompt_id'] for c in chunks if c.get('chunk_type') == 'prompt'))

    # Randomly sample prompts (same prompts across all levels for consistency)
    import random
    random.seed(42)  # Fixed seed for reproducibility
    prompt_ids = random.sample(all_prompt_ids, min(num_prompts, len(all_prompt_ids)))

    print(f"\nTesting {len(prompt_ids)} prompts (out of {len(all_prompt_ids)} total) across {len(redundancy_levels)} redundancy levels")

    # Initialize strategies
    strategy_topk = create_retrieval_strategy('naive')
    strategy_mmr = create_retrieval_strategy('mmr', lambda_param=0.5)
    strategy_qubo = create_retrieval_strategy('qubo', alpha=0.05, penalty=1000.0,
                                               solver='orbit', solver_preset='balanced')

    all_results = []

    for level in redundancy_levels:
        print(f"\n{'='*60}")
        print(f"Testing Redundancy Level {level}")
        print('='*60)

        topk_recalls = []
        mmr_recalls = []
        qubo_recalls = []

        for prompt_id in tqdm(prompt_ids, desc=f"Level {level}", unit="prompt"):
            # Get prompt chunk (by prompt_id field, not chunk_id)
            prompt_chunks = [c for c in chunks if c.get('chunk_type') == 'prompt' and c.get('prompt_id') == prompt_id]
            if not prompt_chunks:
                continue
            prompt_chunk_id = prompt_chunks[0]['chunk_id']  # Get the actual chunk_id for embedding lookup
            prompt_embedding = embeddings_dict.get(prompt_chunk_id)
            if prompt_embedding is None:
                continue

            # Get candidates for this prompt at this redundancy level
            candidates, gold_base_aspects = filter_chunks_for_prompt(
                chunks, prompt_id, level
            )

            if len(candidates) < k or len(gold_base_aspects) == 0:
                continue

            # Prepare candidate results format
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
            topk_results, _ = strategy_topk.retrieve(
                prompt_embedding, candidate_results, k=k
            )
            topk_selected = [r.chunk.metadata for r in topk_results]
            topk_recall = compute_aspect_recall(topk_selected, gold_base_aspects)
            topk_recalls.append(topk_recall)

            # Run MMR retrieval
            mmr_results, _ = strategy_mmr.retrieve(
                prompt_embedding, candidate_results, k=k
            )
            mmr_selected = [r.chunk.metadata for r in mmr_results]
            mmr_recall = compute_aspect_recall(mmr_selected, gold_base_aspects)
            mmr_recalls.append(mmr_recall)

            # Run QUBO retrieval
            qubo_results, _ = strategy_qubo.retrieve(
                prompt_embedding, candidate_results, k=k
            )
            qubo_selected = [r.chunk.metadata for r in qubo_results]
            qubo_recall = compute_aspect_recall(qubo_selected, gold_base_aspects)
            qubo_recalls.append(qubo_recall)

        # Aggregate statistics for this level
        level_results = {
            'redundancy_level': level,
            'num_prompts': len(topk_recalls),
            'topk': {
                'mean_recall': float(np.mean(topk_recalls)) if topk_recalls else 0.0,
                'std_recall': float(np.std(topk_recalls)) if topk_recalls else 0.0,
                'median_recall': float(np.median(topk_recalls)) if topk_recalls else 0.0,
                'min_recall': float(np.min(topk_recalls)) if topk_recalls else 0.0,
                'max_recall': float(np.max(topk_recalls)) if topk_recalls else 0.0
            },
            'mmr': {
                'mean_recall': float(np.mean(mmr_recalls)) if mmr_recalls else 0.0,
                'std_recall': float(np.std(mmr_recalls)) if mmr_recalls else 0.0,
                'median_recall': float(np.median(mmr_recalls)) if mmr_recalls else 0.0,
                'min_recall': float(np.min(mmr_recalls)) if mmr_recalls else 0.0,
                'max_recall': float(np.max(mmr_recalls)) if mmr_recalls else 0.0
            },
            'qubo': {
                'mean_recall': float(np.mean(qubo_recalls)) if qubo_recalls else 0.0,
                'std_recall': float(np.std(qubo_recalls)) if qubo_recalls else 0.0,
                'median_recall': float(np.median(qubo_recalls)) if qubo_recalls else 0.0,
                'min_recall': float(np.min(qubo_recalls)) if qubo_recalls else 0.0,
                'max_recall': float(np.max(qubo_recalls)) if qubo_recalls else 0.0
            }
        }

        print(f"Top-K: {level_results['topk']['mean_recall']:.1f}% ± {level_results['topk']['std_recall']:.1f}%")
        print(f"MMR:   {level_results['mmr']['mean_recall']:.1f}% ± {level_results['mmr']['std_recall']:.1f}%")
        print(f"QUBO:  {level_results['qubo']['mean_recall']:.1f}% ± {level_results['qubo']['std_recall']:.1f}%")

        all_results.append(level_results)

    return all_results


def plot_dose_response_curve(all_results, output_path):
    """
    Generate dose-response curve showing aspect recall vs redundancy level.

    Args:
        all_results: List of result dictionaries for each redundancy level
        output_path: Path to save the plot
    """
    redundancy_levels = [r['redundancy_level'] for r in all_results]
    topk_means = [r['topk']['mean_recall'] for r in all_results]
    topk_stds = [r['topk']['std_recall'] for r in all_results]
    mmr_means = [r['mmr']['mean_recall'] for r in all_results]
    mmr_stds = [r['mmr']['std_recall'] for r in all_results]
    qubo_means = [r['qubo']['mean_recall'] for r in all_results]
    qubo_stds = [r['qubo']['std_recall'] for r in all_results]

    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot lines with error bands
    ax.plot(redundancy_levels, topk_means, 'o-', linewidth=2.5, markersize=8,
            label='Top-K (Baseline)', color='#e74c3c')
    ax.fill_between(redundancy_levels,
                     np.array(topk_means) - np.array(topk_stds),
                     np.array(topk_means) + np.array(topk_stds),
                     alpha=0.2, color='#e74c3c')

    ax.plot(redundancy_levels, mmr_means, '^-', linewidth=2.5, markersize=8,
            label='MMR (SOTA)', color='#f39c12')
    ax.fill_between(redundancy_levels,
                     np.array(mmr_means) - np.array(mmr_stds),
                     np.array(mmr_means) + np.array(mmr_stds),
                     alpha=0.2, color='#f39c12')

    ax.plot(redundancy_levels, qubo_means, 's-', linewidth=2.5, markersize=8,
            label='QUBO (Ours)', color='#3498db')
    ax.fill_between(redundancy_levels,
                     np.array(qubo_means) - np.array(qubo_stds),
                     np.array(qubo_means) + np.array(qubo_stds),
                     alpha=0.2, color='#3498db')

    # Formatting
    ax.set_xlabel('Redundancy Level', fontsize=13, fontweight='bold')
    ax.set_ylabel('Aspect Recall (%)', fontsize=13, fontweight='bold')
    ax.set_title('Experiment 1.3: Dose-Response Curve\nTop-K vs MMR vs QUBO',
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(redundancy_levels)
    ax.set_xticklabels([f'L{i}' for i in redundancy_levels])
    ax.legend(loc='best', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, 105)

    # Add reference lines
    ax.axhline(y=90, color='green', linestyle='--', alpha=0.4, linewidth=1.5)
    ax.axhline(y=50, color='orange', linestyle='--', alpha=0.4, linewidth=1.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")


def analyze_dose_response(all_results):
    """
    Analyze the dose-response relationship.

    Args:
        all_results: List of result dictionaries

    Returns:
        analysis: Dictionary with degradation statistics
    """
    baseline_topk = all_results[0]['topk']['mean_recall']  # Level 0
    baseline_mmr = all_results[0]['mmr']['mean_recall']  # Level 0
    baseline_qubo = all_results[0]['qubo']['mean_recall']  # Level 0

    # Calculate degradation from baseline
    topk_degradations = []
    qubo_degradations = []

    for result in all_results:
        topk_drop = baseline_topk - result['topk']['mean_recall']
        qubo_drop = baseline_qubo - result['qubo']['mean_recall']
        topk_degradations.append(topk_drop)
        qubo_degradations.append(qubo_drop)

    # Check level 1 drop for Top-K
    level_1_topk_drop = topk_degradations[1] if len(topk_degradations) > 1 else 0

    analysis = {
        'baseline_topk': baseline_topk,
        'baseline_mmr': baseline_mmr,
        'baseline_qubo': baseline_qubo,
        'topk_degradations': topk_degradations,
        'qubo_degradations': qubo_degradations,
        'level_1_topk_drop': level_1_topk_drop,
        'max_qubo_variation': max(qubo_degradations) - min(qubo_degradations)
    }

    return analysis


def main():
    """Run Experiment 1.3: Dose-Response Curve."""
    import argparse
    parser = argparse.ArgumentParser(description='Experiment 1.3: Dose-Response Curve')
    parser.add_argument('--num-prompts', type=int, default=10,
                        help='Number of prompts to test (default: 10, use 100 for full experiment)')
    args = parser.parse_args()

    print("="*80)
    print("EXPERIMENT 1.3: DOSE-RESPONSE CURVE")
    print("="*80)
    print(f"Testing with {args.num_prompts} prompts (use --num-prompts 100 for full run)")
    print("="*80)

    # Load data
    chunks, embeddings_dict = load_wikipedia_data()

    # Test redundancy levels 0-5
    redundancy_levels = [0, 1, 2, 3, 5]

    # Run experiment
    all_results = run_dose_response_experiment(chunks, embeddings_dict, redundancy_levels, k=5, num_prompts=args.num_prompts)

    # Save results
    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)

    output_json = results_dir / 'exp_1_3_dose_response_curve.json'
    with open(output_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_json}")

    # Generate plot
    output_plot = results_dir / 'exp_1_3_dose_response_curve.png'
    plot_dose_response_curve(all_results, output_plot)

    # Analyze dose-response
    analysis = analyze_dose_response(all_results)

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Baseline (Level 0):")
    print(f"  Top-K: {analysis['baseline_topk']:.1f}%")
    print(f"  QUBO:  {analysis['baseline_qubo']:.1f}%")
    print(f"\nDegradation from Baseline:")
    for i, level in enumerate(redundancy_levels):
        mmr_drop = analysis['baseline_mmr'] - all_results[i]['mmr']['mean_recall'] if i < len(all_results) else 0
        print(f"  Level {level}: Top-K -{analysis['topk_degradations'][i]:.1f}%, "
              f"MMR -{mmr_drop:.1f}%, QUBO -{analysis['qubo_degradations'][i]:.1f}%")

    # Success criteria check
    print("\n" + "="*80)
    print("SUCCESS CRITERIA")
    print("="*80)

    topk_drops_early = analysis['level_1_topk_drop'] > 20
    qubo_stable = analysis['max_qubo_variation'] <= 5

    print(f"✓ Top-K drops >20% at level 1: {analysis['level_1_topk_drop']:.1f}% - "
          f"{'PASS' if topk_drops_early else 'FAIL'}")
    print(f"✓ QUBO remains flat (±5%): {analysis['max_qubo_variation']:.1f}% variation - "
          f"{'PASS' if qubo_stable else 'FAIL'}")

    overall_success = topk_drops_early and qubo_stable
    print(f"\nOverall: {'✓ SUCCESS' if overall_success else '✗ FAIL'}")

    if overall_success:
        print("\nConclusion: QUBO shows robust dose-response stability while Top-K degrades continuously.")
    else:
        print("\nNote: Results do not fully meet success criteria.")


if __name__ == '__main__':
    main()
