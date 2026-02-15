"""
Experiment 1.1: Poisoned Stress Test

Purpose: Demonstrate that QUBO retrieval maintains aspect recall under redundancy
         while Top-K degrades catastrophically.

Hypothesis: Top-K retrieves duplicates of the same aspect and misses other aspects.
            QUBO rejects duplicates and retrieves distinct aspects.

Method:
1. For each redundancy level (0-5):
   - Load prompts with corresponding redundant chunks
   - For each prompt:
     * Run Top-K retrieval (baseline)
     * Run QUBO retrieval (diversity-aware)
     * Measure aspect recall: # distinct aspects retrieved / 5
   - Aggregate statistics across all prompts

Dataset: Wikipedia dataset with 100 prompts, varying redundancy levels
- Redundancy level 0: 5 gold base + 25 noise (30 total)
- Redundancy level 1: 5 gold base + 5 redundant + 25 noise (35 total)
- Redundancy level 2: 5 gold base + 10 redundant + 25 noise (40 total)
- Redundancy level 3: 5 gold base + 15 redundant + 25 noise (45 total)
- Redundancy level 5: 5 gold base + 25 redundant + 25 noise (55 total)

Settings:
- K = 5 (retrieve 5 chunks)
- QUBO parameters: α = 0.05 (diversity weight), P = 1000 (cardinality penalty)
- QUBO solver: ORBIT with 'balanced' preset

Success Metrics:
- QUBO Aspect Recall: >90% across all redundancy levels
- Top-K Aspect Recall: <30% at redundancy level 5
- QUBO maintains stable performance while Top-K degrades

Output:
- results/exp_1_1_poisoned_stress_test.json (raw results)
- results/exp_1_1_poisoned_stress_test.png (visualization)
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
from collections import defaultdict

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
        redundancy_level: Maximum redundancy level to include (0 = base only)

    Returns:
        candidates: List of candidate chunks (gold base + redundant + noise)
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
            # Include redundant chunks up to redundancy_level
            # redundancy_index: 0 = first redundant, 1 = second, etc.
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
        num_aspects_retrieved: Count of distinct aspects (0-5)
    """
    retrieved_aspects = set()
    for chunk in selected_chunks:
        aspect_id = chunk.get('aspect_id', -1)
        if aspect_id >= 0:  # Valid aspect (not noise)
            retrieved_aspects.add(aspect_id)

    num_gold_aspects = len(gold_base_aspects)
    num_retrieved = len(retrieved_aspects & gold_base_aspects)

    if num_gold_aspects == 0:
        return 0.0, 0

    aspect_recall = 100.0 * num_retrieved / num_gold_aspects
    return aspect_recall, num_retrieved


def run_retrieval_comparison(chunks, embeddings_dict, redundancy_level, k=5, num_prompts=10):
    """
    Run Top-K vs MMR vs QUBO comparison at a specific redundancy level.

    Args:
        chunks: All chunks
        embeddings_dict: All embeddings
        redundancy_level: Redundancy level to test
        k: Number of chunks to retrieve
        num_prompts: Number of prompts to test (default: 10 for fast runs, use 100 for full)

    Returns:
        results: Dictionary with aspect recall statistics for all methods
    """
    # Get all unique prompt IDs (extract prompt_id field from prompt chunks)
    all_prompt_ids = list(set(c['prompt_id'] for c in chunks if c.get('chunk_type') == 'prompt'))

    # Randomly sample prompts
    import random
    random.seed(42)  # Fixed seed for reproducibility
    prompt_ids = random.sample(all_prompt_ids, min(num_prompts, len(all_prompt_ids)))

    print(f"\nRedundancy Level {redundancy_level}: Testing {len(prompt_ids)} prompts (out of {len(all_prompt_ids)} total)")

    # Initialize strategies
    strategy_topk = create_retrieval_strategy('naive')
    strategy_mmr = create_retrieval_strategy('mmr', lambda_param=0.85)
    strategy_qubo = create_retrieval_strategy('qubo', alpha=0.05, penalty=10,
                                               solver='gurobi', solver_preset='balanced')

    topk_recalls = []
    mmr_recalls = []
    qubo_recalls = []
    topk_aspects_counts = []
    mmr_aspects_counts = []
    qubo_aspects_counts = []
    qubo_n_selected = []  # Track number of items selected by QUBO

    # Flag to show detailed output for first prompt only
    show_detailed_output = True

    for idx, prompt_id in enumerate(tqdm(prompt_ids, desc=f"Level {redundancy_level}", unit="prompt")):
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
            chunks, prompt_id, redundancy_level
        )

        if len(candidates) < k or len(gold_base_aspects) == 0:
            continue

        # Show detailed output for first prompt only
        if show_detailed_output and idx == 0:
            print(f"\n{'='*100}")
            print(f"DETAILED OUTPUT FOR FIRST PROMPT (Redundancy Level {redundancy_level})")
            print(f"{'='*100}")
            print(f"\n📝 PROMPT TEXT:")
            print(f"   {prompt_chunks[0]['text'][:200]}...")
            print(f"\n🎯 GOLD BASE ASPECTS ({len(gold_base_aspects)} aspects):")
            gold_base_chunks = [c for c in candidates if c.get('chunk_type') == 'gold_base']
            for aspect_id in sorted(gold_base_aspects):
                aspect_chunks = [c for c in gold_base_chunks if c.get('aspect_id') == aspect_id]
                if aspect_chunks:
                    chunk = aspect_chunks[0]
                    print(f"\n   Aspect {aspect_id} ({chunk.get('aspect_name', 'Unknown')}):")
                    print(f"   {chunk['text'][:150]}...")
            print(f"\n📊 CANDIDATE POOL SIZE: {len(candidates)} chunks")
            print(f"   - Gold base: {len([c for c in candidates if c.get('chunk_type') == 'gold_base'])}")
            print(f"   - Gold redundant: {len([c for c in candidates if c.get('chunk_type') == 'gold_redundant'])}")
            print(f"   - Noise: {len([c for c in candidates if c.get('chunk_type') == 'noise'])}")
            print()

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

        # Sort by similarity (for baseline)
        candidate_results.sort(key=lambda x: x['score'], reverse=True)

        # Run Top-K retrieval
        topk_results, _ = strategy_topk.retrieve(
            prompt_embedding, candidate_results, k=k
        )
        topk_selected = [r.chunk.metadata for r in topk_results]
        topk_recall, topk_count = compute_aspect_recall(topk_selected, gold_base_aspects)
        topk_recalls.append(topk_recall)
        topk_aspects_counts.append(topk_count)

        # Show Top-K results for first prompt
        if show_detailed_output and idx == 0:
            print(f"🔴 TOP-K RESULTS:")
            for rank, result in enumerate(topk_results, 1):
                chunk = result.chunk.metadata
                chunk_type = chunk.get('chunk_type', 'unknown')
                aspect_id = chunk.get('aspect_id', -1)
                aspect_name = chunk.get('aspect_name', 'N/A')
                redundancy_idx = chunk.get('redundancy_index', -1)
                similarity = result.score

                type_emoji = "✓" if chunk_type == "gold_base" else ("↻" if chunk_type == "gold_redundant" else "✗")
                redundancy_str = f"[redundancy={redundancy_idx}]" if chunk_type == "gold_redundant" else ""

                print(f"   {rank}. {type_emoji} Aspect {aspect_id} ({aspect_name}) | {chunk_type} {redundancy_str} | sim={similarity:.3f}")
                print(f"      {chunk['text'][:100]}...")
            print(f"   → Aspect Recall: {topk_recall:.0f}% ({topk_count}/{len(gold_base_aspects)} aspects)")

        # Run MMR retrieval
        mmr_results, _ = strategy_mmr.retrieve(
            prompt_embedding, candidate_results, k=k
        )
        mmr_selected = [r.chunk.metadata for r in mmr_results]
        mmr_recall, mmr_count = compute_aspect_recall(mmr_selected, gold_base_aspects)
        mmr_recalls.append(mmr_recall)
        mmr_aspects_counts.append(mmr_count)

        # Show MMR results for first prompt
        if show_detailed_output and idx == 0:
            print(f"\n🟠 MMR RESULTS:")
            for rank, result in enumerate(mmr_results, 1):
                chunk = result.chunk.metadata
                chunk_type = chunk.get('chunk_type', 'unknown')
                aspect_id = chunk.get('aspect_id', -1)
                aspect_name = chunk.get('aspect_name', 'N/A')
                redundancy_idx = chunk.get('redundancy_index', -1)
                similarity = result.score

                type_emoji = "✓" if chunk_type == "gold_base" else ("↻" if chunk_type == "gold_redundant" else "✗")
                redundancy_str = f"[redundancy={redundancy_idx}]" if chunk_type == "gold_redundant" else ""

                print(f"   {rank}. {type_emoji} Aspect {aspect_id} ({aspect_name}) | {chunk_type} {redundancy_str} | sim={similarity:.3f}")
                print(f"      {chunk['text'][:100]}...")
            print(f"   → Aspect Recall: {mmr_recall:.0f}% ({mmr_count}/{len(gold_base_aspects)} aspects)")

        # Run QUBO retrieval
        qubo_results, _ = strategy_qubo.retrieve(
            prompt_embedding, candidate_results, k=k
        )
        qubo_selected = [r.chunk.metadata for r in qubo_results]
        qubo_recall, qubo_count = compute_aspect_recall(qubo_selected, gold_base_aspects)
        qubo_recalls.append(qubo_recall)
        qubo_aspects_counts.append(qubo_count)
        qubo_n_selected.append(len(qubo_selected))  # Track number of items selected by QUBO

        # Show QUBO results for first prompt
        if show_detailed_output and idx == 0:
            print(f"\n🔵 QUBO RESULTS:")
            for rank, result in enumerate(qubo_results, 1):
                chunk = result.chunk.metadata
                chunk_type = chunk.get('chunk_type', 'unknown')
                aspect_id = chunk.get('aspect_id', -1)
                aspect_name = chunk.get('aspect_name', 'N/A')
                redundancy_idx = chunk.get('redundancy_index', -1)
                similarity = result.score

                type_emoji = "✓" if chunk_type == "gold_base" else ("↻" if chunk_type == "gold_redundant" else "✗")
                redundancy_str = f"[redundancy={redundancy_idx}]" if chunk_type == "gold_redundant" else ""

                print(f"   {rank}. {type_emoji} Aspect {aspect_id} ({aspect_name}) | {chunk_type} {redundancy_str} | sim={similarity:.3f}")
                print(f"      {chunk['text'][:100]}...")
            print(f"   → Aspect Recall: {qubo_recall:.0f}% ({qubo_count}/{len(gold_base_aspects)} aspects)")
            print(f"   → Selected: {len(qubo_selected)} chunks (target: {k})")
            print(f"{'='*100}\n")

    # Aggregate statistics
    results = {
        'redundancy_level': redundancy_level,
        'num_prompts': len(topk_recalls),
        'topk': {
            'mean_aspect_recall': float(np.mean(topk_recalls)) if topk_recalls else 0.0,
            'std_aspect_recall': float(np.std(topk_recalls)) if topk_recalls else 0.0,
            'mean_aspects_retrieved': float(np.mean(topk_aspects_counts)) if topk_aspects_counts else 0.0,
            'perfect_recalls': int(np.sum(np.array(topk_recalls) == 100.0)) if topk_recalls else 0
        },
        'mmr': {
            'mean_aspect_recall': float(np.mean(mmr_recalls)) if mmr_recalls else 0.0,
            'std_aspect_recall': float(np.std(mmr_recalls)) if mmr_recalls else 0.0,
            'mean_aspects_retrieved': float(np.mean(mmr_aspects_counts)) if mmr_aspects_counts else 0.0,
            'perfect_recalls': int(np.sum(np.array(mmr_recalls) == 100.0)) if mmr_recalls else 0
        },
        'qubo': {
            'mean_aspect_recall': float(np.mean(qubo_recalls)) if qubo_recalls else 0.0,
            'std_aspect_recall': float(np.std(qubo_recalls)) if qubo_recalls else 0.0,
            'mean_aspects_retrieved': float(np.mean(qubo_aspects_counts)) if qubo_aspects_counts else 0.0,
            'perfect_recalls': int(np.sum(np.array(qubo_recalls) == 100.0)) if qubo_recalls else 0,
            'mean_n_selected': float(np.mean(qubo_n_selected)) if qubo_n_selected else 0.0,
            'std_n_selected': float(np.std(qubo_n_selected)) if qubo_n_selected else 0.0
        }
    }

    print(f"  Top-K: {results['topk']['mean_aspect_recall']:.1f}% ± {results['topk']['std_aspect_recall']:.1f}%")
    print(f"  MMR:   {results['mmr']['mean_aspect_recall']:.1f}% ± {results['mmr']['std_aspect_recall']:.1f}%")
    print(f"  QUBO:  {results['qubo']['mean_aspect_recall']:.1f}% ± {results['qubo']['std_aspect_recall']:.1f}%")
    print(f"         (Selected: {results['qubo']['mean_n_selected']:.2f} ± {results['qubo']['std_n_selected']:.2f} chunks)")

    return results


def plot_results(all_results, output_path):
    """
    Generate visualization comparing Top-K vs MMR vs QUBO across redundancy levels.

    Args:
        all_results: List of result dictionaries for each redundancy level
        output_path: Path to save the plot
    """
    redundancy_levels = [r['redundancy_level'] for r in all_results]
    topk_means = [r['topk']['mean_aspect_recall'] for r in all_results]
    topk_stds = [r['topk']['std_aspect_recall'] for r in all_results]
    mmr_means = [r['mmr']['mean_aspect_recall'] for r in all_results]
    mmr_stds = [r['mmr']['std_aspect_recall'] for r in all_results]
    qubo_means = [r['qubo']['mean_aspect_recall'] for r in all_results]
    qubo_stds = [r['qubo']['std_aspect_recall'] for r in all_results]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.array(redundancy_levels)
    width = 0.25

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
    ax.set_title('Experiment 1.1: Poisoned Stress Test\nTop-K vs MMR vs QUBO',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{i}' for i in redundancy_levels])
    ax.legend(loc='best', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 105)

    # Add horizontal reference line at 90%
    ax.axhline(y=90, color='green', linestyle='--', alpha=0.5, linewidth=1.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")


def main():
    """Run Experiment 1.1: Poisoned Stress Test."""
    import argparse
    parser = argparse.ArgumentParser(description='Experiment 1.1: Poisoned Stress Test')
    parser.add_argument('--num-prompts', type=int, default=10,
                        help='Number of prompts to test (default: 10, use 100 for full experiment)')
    args = parser.parse_args()

    print("="*80)
    print("EXPERIMENT 1.1: POISONED STRESS TEST")
    print("="*80)
    print(f"Testing with {args.num_prompts} prompts (use --num-prompts 100 for full run)")
    print("="*80)

    # Load data
    chunks, embeddings_dict = load_wikipedia_data()

    # Test redundancy levels 0-5
    redundancy_levels = [0, 1, 2, 3, 4, 5]
    all_results = []

    for level in redundancy_levels:
        results = run_retrieval_comparison(chunks, embeddings_dict, level, k=5, num_prompts=args.num_prompts)
        all_results.append(results)

    # Save results
    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)

    output_json = results_dir / 'exp_1_1_poisoned_stress_test.json'
    with open(output_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_json}")

    # Generate plot
    output_plot = results_dir / 'exp_1_1_poisoned_stress_test.png'
    plot_results(all_results, output_plot)

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for result in all_results:
        level = result['redundancy_level']
        topk_recall = result['topk']['mean_aspect_recall']
        mmr_recall = result['mmr']['mean_aspect_recall']
        qubo_recall = result['qubo']['mean_aspect_recall']
        print(f"Level {level}: Top-K = {topk_recall:.1f}%, MMR = {mmr_recall:.1f}%, QUBO = {qubo_recall:.1f}%")

    # Success criteria check
    print("\n" + "="*80)
    print("SUCCESS CRITERIA")
    print("="*80)

    level_5_result = [r for r in all_results if r['redundancy_level'] == 5][0]
    qubo_min = min(r['qubo']['mean_aspect_recall'] for r in all_results)
    topk_at_5 = level_5_result['topk']['mean_aspect_recall']

    qubo_success = qubo_min > 90
    topk_success = topk_at_5 < 30

    print(f"✓ QUBO maintains >90% recall: {qubo_min:.1f}% (min) - {'PASS' if qubo_success else 'FAIL'}")
    print(f"✓ Top-K drops to <30% at level 5: {topk_at_5:.1f}% - {'PASS' if topk_success else 'FAIL'}")

    overall_success = qubo_success and topk_success
    print(f"\nOverall: {'✓ SUCCESS' if overall_success else '✗ FAIL'}")


if __name__ == '__main__':
    main()
