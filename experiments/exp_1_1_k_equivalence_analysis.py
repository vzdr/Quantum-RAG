"""
Experiment 1.1: K-Equivalence Analysis

Purpose: Determine how much k must increase for Top-K/MMR to match QUBO's aspect recall.
         Tests incrementally higher k values until performance is within 3% of QUBO.
"""
import sys
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add project root to path
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from core.retrieval import NaiveRetrieval, MMRRetrieval
from core.utils import (
    load_wikipedia_dataset,
    filter_chunks_by_prompt,
    get_prompt_embedding,
    compute_aspect_recall
)

def estimate_avg_chunk_length(chunks):
    """Estimate average chunk length in tokens (approximate as words / 0.75)."""
    sample_size = min(1000, len(chunks))
    sample_chunks = np.random.choice(chunks, size=sample_size, replace=False)
    lengths = [len(c['text'].split()) / 0.75 for c in sample_chunks]  # rough token estimate
    return np.mean(lengths)

def test_k_value(strategy, chunks, embeddings_dict, redundancy_level, k, num_prompts=50):
    """Test a specific k value and return average aspect recall."""
    all_prompt_ids = list(set(c['prompt_id'] for c in chunks if c.get('chunk_type') == 'prompt'))
    prompt_ids = np.random.choice(all_prompt_ids, size=min(num_prompts, len(all_prompt_ids)), replace=False).tolist()

    recalls = []

    for prompt_id in prompt_ids:
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

        selected = strategy.retrieve(prompt_embedding, candidate_results, k=k)
        selected_meta = [r.chunk.metadata for r in selected]
        recall, _ = compute_aspect_recall(selected_meta, gold_aspects)
        recalls.append(recall)

    return np.mean(recalls) if recalls else 0.0

def find_equivalent_k(method_name, baseline_recall, chunks, embeddings_dict, redundancy_level,
                      starting_k=5, max_k=50, tolerance=3.0):
    """
    Find the minimum k where method achieves within tolerance% of baseline_recall.
    Returns (k_needed, avg_recall) or (None, current_recall) if already within tolerance.
    """
    if method_name == 'topk':
        strategy = NaiveRetrieval()
    else:  # mmr
        strategy = MMRRetrieval(lambda_param=0.85)

    # Check if starting k is already sufficient
    current_recall = test_k_value(strategy, chunks, embeddings_dict, redundancy_level, starting_k)

    if current_recall >= baseline_recall - tolerance:
        return None, current_recall  # Already within tolerance

    # Binary search would be faster, but linear is simpler and more transparent
    print(f"    Testing {method_name.upper()} with increasing k...", end='', flush=True)

    for k in range(starting_k + 1, max_k + 1):
        recall = test_k_value(strategy, chunks, embeddings_dict, redundancy_level, k)
        print(f" k={k}({recall:.1f}%)", end='', flush=True)

        if recall >= baseline_recall - tolerance:
            print()  # newline
            return k, recall

    print(f" [max k={max_k} reached]")
    return max_k, test_k_value(strategy, chunks, embeddings_dict, redundancy_level, max_k)

def main():
    print("--- Running Experiment 1.1: K-Equivalence Analysis ---")

    # Load baseline results from exp_1
    results_dir = project_root / 'results'
    baseline_file = results_dir / 'exp_1_poisoned_stress_test.json'

    if not baseline_file.exists():
        print(f"ERROR: Baseline results not found at {baseline_file}")
        print("Please run exp_1_poisoned_stress_test.py first.")
        return

    with open(baseline_file, 'r') as f:
        baseline_results = json.load(f)

    print(f"Loaded baseline results from {baseline_file}")

    # Load dataset
    print("Loading Wikipedia dataset...")
    chunks, embeddings_dict = load_wikipedia_dataset()
    avg_chunk_tokens = estimate_avg_chunk_length(chunks)
    print(f"Average chunk length: ~{avg_chunk_tokens:.0f} tokens")

    # Run analysis for each redundancy level
    analysis_results = []
    starting_k = 5

    for result in baseline_results:
        level = result['redundancy_level']
        qubo_recall = result['qubo']['mean_aspect_recall']
        topk_recall = result['topk']['mean_aspect_recall']
        mmr_recall = result['mmr']['mean_aspect_recall']

        print(f"\n--- Level {level} (QUBO baseline: {qubo_recall:.1f}%) ---")

        level_result = {
            'redundancy_level': level,
            'qubo_recall': qubo_recall,
            'baseline_k': starting_k,
        }

        # Analyze Top-K
        print(f"  Top-K (current: {topk_recall:.1f}%)")
        if topk_recall >= qubo_recall - 3.0:
            print(f"    ✓ Already within 3% - no increase needed")
            level_result['topk'] = {
                'k_needed': starting_k,
                'k_increase': 0,
                'recall_achieved': topk_recall,
                'extra_tokens': 0
            }
        else:
            k_needed, recall_achieved = find_equivalent_k(
                'topk', qubo_recall, chunks, embeddings_dict, level, starting_k
            )
            k_increase = k_needed - starting_k if k_needed else 0
            extra_tokens = int(k_increase * avg_chunk_tokens)
            level_result['topk'] = {
                'k_needed': k_needed,
                'k_increase': k_increase,
                'recall_achieved': recall_achieved,
                'extra_tokens': extra_tokens
            }
            print(f"    → Needs k={k_needed} (+{k_increase}) for {recall_achieved:.1f}% recall")
            print(f"    → Extra context: ~{extra_tokens:,} tokens")

        # Analyze MMR
        print(f"  MMR (current: {mmr_recall:.1f}%)")
        if mmr_recall >= qubo_recall - 3.0:
            print(f"    ✓ Already within 3% - no increase needed")
            level_result['mmr'] = {
                'k_needed': starting_k,
                'k_increase': 0,
                'recall_achieved': mmr_recall,
                'extra_tokens': 0
            }
        else:
            k_needed, recall_achieved = find_equivalent_k(
                'mmr', qubo_recall, chunks, embeddings_dict, level, starting_k
            )
            k_increase = k_needed - starting_k if k_needed else 0
            extra_tokens = int(k_increase * avg_chunk_tokens)
            level_result['mmr'] = {
                'k_needed': k_needed,
                'k_increase': k_increase,
                'recall_achieved': recall_achieved,
                'extra_tokens': extra_tokens
            }
            print(f"    → Needs k={k_needed} (+{k_increase}) for {recall_achieved:.1f}% recall")
            print(f"    → Extra context: ~{extra_tokens:,} tokens")

        analysis_results.append(level_result)

    # Save results
    output_file = results_dir / 'exp_1_1_k_equivalence_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    print(f"\n✓ Results saved to {output_file}")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY: K-Increase Required to Match QUBO Performance")
    print("="*80)
    print(f"{'Level':<8} {'Top-K k':<12} {'Top-K Tokens':<15} {'MMR k':<12} {'MMR Tokens':<15}")
    print("-"*80)

    for r in analysis_results:
        level = r['redundancy_level']
        topk_k = f"{r['baseline_k']}→{r['topk']['k_needed']}" if r['topk']['k_increase'] > 0 else f"{r['baseline_k']} ✓"
        topk_tokens = f"+{r['topk']['extra_tokens']:,}" if r['topk']['k_increase'] > 0 else "0"
        mmr_k = f"{r['baseline_k']}→{r['mmr']['k_needed']}" if r['mmr']['k_increase'] > 0 else f"{r['baseline_k']} ✓"
        mmr_tokens = f"+{r['mmr']['extra_tokens']:,}" if r['mmr']['k_increase'] > 0 else "0"

        print(f"L{level:<7} {topk_k:<12} {topk_tokens:<15} {mmr_k:<12} {mmr_tokens:<15}")

    print("="*80)
    print("✓ Within 3% means no k increase needed")
    print("--- Experiment Complete ---")

if __name__ == '__main__':
    main()
