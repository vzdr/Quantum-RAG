"""
Experiment 1.1: The Poisoned Stress Test

Purpose: Prove Top-K fails catastrophically under redundancy, while QUBO and MMR
rescue distinct facts.

Hypothesis: Top-K retrieves duplicates of the same aspect and misses other aspects.
QUBO and MMR reject duplicates and retrieve distinct aspects.

Method:
1. Test three retrieval methods: Top-K, MMR, QUBO
2. For each redundancy level (0-5):
   - Run all methods on 100 prompts
   - Measure: Aspect Recall, Gold Recall, Redundancy Rate, Redundancy Score
3. Generate bar charts comparing methods across redundancy levels

Dataset: Wikipedia dataset, 100 prompts, K=5

Settings:
- Redundancy levels: 0-5
- K = 5 (context window)
- N = 100 (retrieval pool)
- Alpha = 0.7 (relevance weight for QUBO)
- Beta = 0.05 (diversity weight for QUBO)
- Lambda = 0.7 (diversity weight for MMR)

Success Metrics:
- QUBO Aspect Recall: >90%
- Top-K Aspect Recall: <30%
- QUBO Redundancy: <0.1
- Top-K Redundancy: >0.8
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
from collections import defaultdict
from tqdm import tqdm

from core.qubo_solver import solve_diverse_retrieval_qubo


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

    Returns:
        gold_chunks: List of gold chunks (base + redundant up to level)
        noise_chunks: List of noise chunks
        distinct_aspects: Set of distinct aspect IDs
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
            if redundancy_idx < redundancy_level:
                gold_chunks.append(chunk)
        elif chunk_type == 'noise':
            noise_chunks.append(chunk)

    return gold_chunks, noise_chunks, distinct_aspects


def retrieve_topk(query_embedding, candidate_chunks, embeddings_dict, k=5):
    """
    Top-K retrieval: Select top K chunks by cosine similarity to query.

    Returns:
        selected_indices: List of selected chunk indices
    """
    # Compute similarities
    similarities = []
    for i, chunk in enumerate(candidate_chunks):
        chunk_id = chunk['chunk_id']
        if chunk_id not in embeddings_dict:
            similarities.append(0.0)
            continue

        chunk_emb = embeddings_dict[chunk_id]
        sim = np.dot(query_embedding, chunk_emb) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(chunk_emb) + 1e-10
        )
        similarities.append(sim)

    # Select top K
    top_indices = np.argsort(similarities)[::-1][:k]
    return top_indices.tolist()


def retrieve_mmr(query_embedding, candidate_chunks, embeddings_dict, k=5, lambda_param=0.7):
    """
    MMR (Maximal Marginal Relevance) retrieval: Balance relevance and diversity.

    MMR = λ * Sim(chunk, query) - (1-λ) * max(Sim(chunk, selected))

    Returns:
        selected_indices: List of selected chunk indices
    """
    # Precompute similarities to query
    query_sims = []
    chunk_embeddings = []

    for chunk in candidate_chunks:
        chunk_id = chunk['chunk_id']
        if chunk_id not in embeddings_dict:
            query_sims.append(0.0)
            chunk_embeddings.append(np.zeros_like(query_embedding))
            continue

        chunk_emb = embeddings_dict[chunk_id]
        chunk_embeddings.append(chunk_emb)

        sim = np.dot(query_embedding, chunk_emb) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(chunk_emb) + 1e-10
        )
        query_sims.append(sim)

    chunk_embeddings = np.array(chunk_embeddings)
    query_sims = np.array(query_sims)

    # MMR selection
    selected_indices = []
    remaining_indices = list(range(len(candidate_chunks)))

    # Select first chunk (highest similarity)
    first_idx = int(np.argmax(query_sims))
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)

    # Iteratively select remaining chunks
    for _ in range(k - 1):
        if not remaining_indices:
            break

        mmr_scores = []
        for idx in remaining_indices:
            # Relevance term
            relevance = query_sims[idx]

            # Diversity term: max similarity to already selected
            max_sim = 0.0
            for selected_idx in selected_indices:
                sim = np.dot(chunk_embeddings[idx], chunk_embeddings[selected_idx]) / (
                    np.linalg.norm(chunk_embeddings[idx]) * np.linalg.norm(chunk_embeddings[selected_idx]) + 1e-10
                )
                max_sim = max(max_sim, sim)

            # MMR score
            mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
            mmr_scores.append(mmr)

        # Select chunk with highest MMR score
        best_idx = remaining_indices[int(np.argmax(mmr_scores))]
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

    return selected_indices


def retrieve_qubo(query_embedding, candidate_chunks, embeddings_dict, k=5, alpha=0.05):
    """
    QUBO diversity-aware retrieval.

    Returns:
        selected_indices: List of selected chunk indices
    """
    # Get embeddings for all candidates
    chunk_embeddings = []
    valid_indices = []

    for i, chunk in enumerate(candidate_chunks):
        chunk_id = chunk['chunk_id']
        if chunk_id in embeddings_dict:
            chunk_embeddings.append(embeddings_dict[chunk_id])
            valid_indices.append(i)

    if len(chunk_embeddings) < k:
        return valid_indices[:k]

    chunk_embeddings = np.array(chunk_embeddings)

    # Solve QUBO using ORBIT solver
    selected_local_indices, metadata = solve_diverse_retrieval_qubo(
        query_embedding=query_embedding,
        candidate_embeddings=chunk_embeddings,
        k=k,
        alpha=alpha,
        solver='orbit'
    )

    # Map back to original indices
    selected_indices = [valid_indices[i] for i in selected_local_indices]
    return selected_indices


def compute_metrics(selected_indices, candidate_chunks, embeddings_dict, distinct_aspects):
    """
    Compute retrieval metrics.

    Returns:
        metrics: Dict with aspect_recall, gold_recall, redundancy_rate, redundancy_score
    """
    # Aspect recall: Are all 5 aspects covered?
    retrieved_aspects = set()
    retrieved_gold_count = 0
    total_gold = 0

    for i, chunk in enumerate(candidate_chunks):
        chunk_type = chunk.get('chunk_type', '')
        aspect_id = chunk.get('aspect_id', -1)

        if chunk_type in ['gold_base', 'gold_redundant']:
            total_gold += 1
            if i in selected_indices:
                retrieved_gold_count += 1
                retrieved_aspects.add(aspect_id)

    aspect_recall = 1.0 if len(retrieved_aspects) == len(distinct_aspects) else 0.0
    gold_recall = retrieved_gold_count / total_gold if total_gold > 0 else 0.0

    # Redundancy rate: What fraction of retrieved chunks share the same aspect?
    aspect_counts = defaultdict(int)
    for idx in selected_indices:
        chunk = candidate_chunks[idx]
        if chunk.get('chunk_type') in ['gold_base', 'gold_redundant']:
            aspect_id = chunk.get('aspect_id', -1)
            aspect_counts[aspect_id] += 1

    redundancy_count = sum(max(0, count - 1) for count in aspect_counts.values())
    redundancy_rate = redundancy_count / len(selected_indices) if len(selected_indices) > 0 else 0.0

    # Redundancy score: Average pairwise similarity in retrieved set
    if len(selected_indices) < 2:
        redundancy_score = 0.0
    else:
        similarities = []
        for i in range(len(selected_indices)):
            for j in range(i + 1, len(selected_indices)):
                idx_i = selected_indices[i]
                idx_j = selected_indices[j]

                chunk_i = candidate_chunks[idx_i]
                chunk_j = candidate_chunks[idx_j]

                emb_i = embeddings_dict.get(chunk_i['chunk_id'])
                emb_j = embeddings_dict.get(chunk_j['chunk_id'])

                if emb_i is not None and emb_j is not None:
                    sim = np.dot(emb_i, emb_j) / (
                        np.linalg.norm(emb_i) * np.linalg.norm(emb_j) + 1e-10
                    )
                    similarities.append(sim)

        redundancy_score = np.mean(similarities) if similarities else 0.0

    return {
        'aspect_recall': aspect_recall,
        'gold_recall': gold_recall,
        'redundancy_rate': redundancy_rate,
        'redundancy_score': float(redundancy_score)
    }


def main():
    print("=" * 80)
    print("EXPERIMENT 1.1: The Poisoned Stress Test")
    print("=" * 80)
    print()

    # Configuration
    REDUNDANCY_LEVELS = [0, 1, 2, 3, 4, 5]
    K = 5
    ALPHA = 0.05  # QUBO diversity weight
    LAMBDA = 0.7  # MMR diversity weight
    MAX_PROMPTS = 10  # Using 10 prompts for faster testing (ORBIT is slow ~12s/prompt)

    print(f"Settings:")
    print(f"  Redundancy Levels: {REDUNDANCY_LEVELS}")
    print(f"  K (chunks to select): {K}")
    print(f"  Max Prompts: {MAX_PROMPTS}")
    print(f"  QUBO Alpha: {ALPHA}")
    print(f"  MMR Lambda: {LAMBDA}")
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
            continue
        prompt_id = chunk.get('prompt_id', '')
        if prompt_id:
            prompts_map[prompt_id].append(chunk)

    print(f"Found {len(prompts_map)} unique prompts")
    print()

    # Results storage
    results_by_level = {}

    # Process each redundancy level
    for level in REDUNDANCY_LEVELS:
        print("=" * 80)
        print(f"Processing Redundancy Level {level}")
        print("=" * 80)

        level_results = {
            'topk': {'aspect_recalls': [], 'gold_recalls': [], 'redundancy_rates': [], 'redundancy_scores': []},
            'mmr': {'aspect_recalls': [], 'gold_recalls': [], 'redundancy_rates': [], 'redundancy_scores': []},
            'qubo': {'aspect_recalls': [], 'gold_recalls': [], 'redundancy_rates': [], 'redundancy_scores': []}
        }

        processed_prompts = 0

        for prompt_id, prompt_chunks_all in list(prompts_map.items())[:MAX_PROMPTS]:
            # Filter to redundancy level
            gold_chunks, noise_chunks, distinct_aspects = filter_chunks_for_prompt(
                prompt_chunks_all, prompt_id, redundancy_level=level
            )

            if len(distinct_aspects) < 3:  # Need reasonable number of aspects
                continue

            candidate_chunks = gold_chunks + noise_chunks

            if len(candidate_chunks) < K:
                continue

            # Use first gold chunk as query approximation
            if len(gold_chunks) > 0:
                query_chunk_id = gold_chunks[0]['chunk_id']
                if query_chunk_id not in embeddings_dict:
                    continue
                query_embedding = embeddings_dict[query_chunk_id]
            else:
                continue

            # Run three retrieval methods
            topk_indices = retrieve_topk(query_embedding, candidate_chunks, embeddings_dict, k=K)
            mmr_indices = retrieve_mmr(query_embedding, candidate_chunks, embeddings_dict, k=K, lambda_param=LAMBDA)
            qubo_indices = retrieve_qubo(query_embedding, candidate_chunks, embeddings_dict, k=K, alpha=ALPHA)

            # Compute metrics
            topk_metrics = compute_metrics(topk_indices, candidate_chunks, embeddings_dict, distinct_aspects)
            mmr_metrics = compute_metrics(mmr_indices, candidate_chunks, embeddings_dict, distinct_aspects)
            qubo_metrics = compute_metrics(qubo_indices, candidate_chunks, embeddings_dict, distinct_aspects)

            # Store results
            for method, metrics in [('topk', topk_metrics), ('mmr', mmr_metrics), ('qubo', qubo_metrics)]:
                level_results[method]['aspect_recalls'].append(metrics['aspect_recall'])
                level_results[method]['gold_recalls'].append(metrics['gold_recall'])
                level_results[method]['redundancy_rates'].append(metrics['redundancy_rate'])
                level_results[method]['redundancy_scores'].append(metrics['redundancy_score'])

            processed_prompts += 1

        print(f"Processed {processed_prompts} prompts at level {level}")

        # Compute averages
        results_by_level[level] = {}
        for method in ['topk', 'mmr', 'qubo']:
            results_by_level[level][method] = {
                'aspect_recall': float(np.mean(level_results[method]['aspect_recalls'])) if level_results[method]['aspect_recalls'] else 0.0,
                'gold_recall': float(np.mean(level_results[method]['gold_recalls'])) if level_results[method]['gold_recalls'] else 0.0,
                'redundancy_rate': float(np.mean(level_results[method]['redundancy_rates'])) if level_results[method]['redundancy_rates'] else 0.0,
                'redundancy_score': float(np.mean(level_results[method]['redundancy_scores'])) if level_results[method]['redundancy_scores'] else 0.0,
                'n_prompts': int(processed_prompts)
            }

        print(f"  Top-K  - Aspect Recall: {results_by_level[level]['topk']['aspect_recall']:.3f}")
        print(f"  MMR    - Aspect Recall: {results_by_level[level]['mmr']['aspect_recall']:.3f}")
        print(f"  QUBO   - Aspect Recall: {results_by_level[level]['qubo']['aspect_recall']:.3f}")
        print()

    # Generate visualizations
    print("=" * 80)
    print("Generating Plots...")
    print("=" * 80)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    metrics = ['aspect_recall', 'gold_recall', 'redundancy_rate', 'redundancy_score']
    titles = ['Aspect Recall', 'Gold Recall', 'Redundancy Rate', 'Redundancy Score']
    ylabels = ['Aspect Recall (%)', 'Gold Recall (%)', 'Redundancy Rate', 'Avg Pairwise Similarity']

    for idx, (metric, title, ylabel) in enumerate(zip(metrics, titles, ylabels)):
        ax = axes[idx // 2, idx % 2]

        topk_vals = [results_by_level[level]['topk'][metric] * 100 if 'recall' in metric else results_by_level[level]['topk'][metric] for level in REDUNDANCY_LEVELS]
        mmr_vals = [results_by_level[level]['mmr'][metric] * 100 if 'recall' in metric else results_by_level[level]['mmr'][metric] for level in REDUNDANCY_LEVELS]
        qubo_vals = [results_by_level[level]['qubo'][metric] * 100 if 'recall' in metric else results_by_level[level]['qubo'][metric] for level in REDUNDANCY_LEVELS]

        x = np.arange(len(REDUNDANCY_LEVELS))
        width = 0.25

        ax.bar(x - width, topk_vals, width, label='Top-K', color='#e74c3c')
        ax.bar(x, mmr_vals, width, label='MMR', color='#f39c12')
        ax.bar(x + width, qubo_vals, width, label='QUBO', color='#27ae60')

        ax.set_xlabel('Redundancy Level', fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(REDUNDANCY_LEVELS)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Exp 1.1: Poisoned Stress Test - Top-K vs MMR vs QUBO', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    # Save plot
    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / 'exp_1_1_poisoned_stress_test.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

    # Save numerical results
    results_json_path = results_dir / 'exp_1_1_results.json'
    with open(results_json_path, 'w') as f:
        json.dump(results_by_level, f, indent=2)
    print(f"Results saved to: {results_json_path}")

    # Print summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for level in REDUNDANCY_LEVELS:
        print(f"\nRedundancy Level {level}:")
        print(f"  Top-K:  Aspect={results_by_level[level]['topk']['aspect_recall']:.1%}, Redundancy={results_by_level[level]['topk']['redundancy_score']:.3f}")
        print(f"  MMR:    Aspect={results_by_level[level]['mmr']['aspect_recall']:.1%}, Redundancy={results_by_level[level]['mmr']['redundancy_score']:.3f}")
        print(f"  QUBO:   Aspect={results_by_level[level]['qubo']['aspect_recall']:.1%}, Redundancy={results_by_level[level]['qubo']['redundancy_score']:.3f}")

    print()
    print("=" * 80)
    print("EXPERIMENT 1.1 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
