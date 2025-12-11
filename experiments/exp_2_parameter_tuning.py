"""
Experiment 2: Parameter Sensitivity Analysis

Purpose: Identify optimal QUBO parameters and validate robustness to parameter choices.
"""
import sys
import json
import argparse
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path and import core modules
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from core.retrieval import Retriever
from core.embedding import EmbeddingGenerator
from core.storage import VectorStore
from core.utils import (
    load_wikipedia_dataset,
    filter_chunks_by_prompt,
    compute_aspect_recall
)

def evaluate_parameters(
    retriever: Retriever, chunks, embeddings_dict, alpha, penalty, prompt_ids, k=5,
    redundancy_levels=range(6)
):
    """
    Evaluates a single (alpha, penalty) parameter combination.
    Returns a results dictionary or None if the parameters are invalid.
    """
    total_recalls = []
    
    for level in redundancy_levels:
        level_recalls = []
        for prompt_id in prompt_ids:
            query_text = ""
            # Find the prompt text
            for chunk in chunks:
                if chunk.get('chunk_type') == 'prompt' and chunk.get('prompt_id') == prompt_id:
                    query_text = chunk['text']
                    break
            
            if not query_text:
                continue

            results = retriever.retrieve(
                query=query_text,
                k=k,
                strategy='qubo',
                alpha=alpha,
                penalty=penalty
            )
            
            # Get gold aspects for evaluation
            _, gold_aspects, _, _, _ = filter_chunks_by_prompt(chunks, prompt_id, level)
            
            # Compute recall
            recall, _ = compute_aspect_recall([r.chunk for r in results], gold_aspects)
            level_recalls.append(recall)

        if not level_recalls: return None
        total_recalls.extend(level_recalls)

    return {
        'alpha': alpha,
        'penalty': penalty,
        'overall_mean_aspect_recall': np.mean(total_recalls),
    }

def grid_search(retriever: Retriever, chunks, embeddings_dict, **kwargs):
    """Performs grid search over alpha and penalty parameters."""
    alpha_range = np.arange(kwargs['alpha_min'], kwargs['alpha_max'], kwargs['alpha_step'])
    penalty_values = kwargs['penalty_values']
    num_prompts = kwargs['num_prompts']
    
    all_prompt_ids = list(set(c['prompt_id'] for c in chunks if c.get('chunk_type') == 'prompt'))
    prompt_ids = np.random.choice(all_prompt_ids, size=min(num_prompts, len(all_prompt_ids)), replace=False).tolist()
    
    print(f"Starting grid search with {len(alpha_range) * len(penalty_values)} combinations...")
    
    results = []
    param_combinations = [(p, a) for p in penalty_values for a in alpha_range]

    for penalty, alpha in tqdm(param_combinations, desc="Grid Search"):
        result = evaluate_parameters(
            retriever, chunks, embeddings_dict, alpha, penalty, prompt_ids, k=kwargs['k']
        )
        if result:
            results.append(result)
            
    results.sort(key=lambda x: x['overall_mean_aspect_recall'], reverse=True)
    return results

def plot_heatmap(results, output_path):
    """Creates and saves a heatmap of the grid search results."""
    if not results:
        print("No valid results to plot.")
        return

    alphas = sorted(list(set(r['alpha'] for r in results)))
    penalties = sorted(list(set(r['penalty'] for r in results)))
    heatmap_data = np.full((len(penalties), len(alphas)), np.nan)

    for r in results:
        # Check if alpha and penalty are in the lists before getting index
        if r['alpha'] in alphas and r['penalty'] in penalties:
            alpha_idx = alphas.index(r['alpha'])
            penalty_idx = penalties.index(r['penalty'])
            heatmap_data[penalty_idx, alpha_idx] = r['overall_mean_aspect_recall']

    plt.figure(figsize=(14, 8))
    sns.heatmap(
        heatmap_data, xticklabels=[f"{a:.2f}" for a in alphas], yticklabels=penalties,
        annot=True, fmt='.1f', cmap='RdYlGn', vmin=0, vmax=100,
        cbar_kws={'label': 'Average Aspect Recall (%)'}
    )
    
    plt.xlabel('Alpha (α) - Diversity Weight', fontweight='bold')
    plt.ylabel('Penalty (P) - Cardinality Constraint', fontweight='bold')
    plt.title('QUBO Parameter Grid Search: Average Aspect Recall', fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Heatmap saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='QUBO Parameter Grid Search')
    parser.add_argument('--alpha-min', type=float, default=0.0)
    parser.add_argument('--alpha-max', type=float, default=0.1)
    parser.add_argument('--alpha-step', type=float, default=0.01)
    parser.add_argument('--penalty-values', type=float, nargs='+', default=[0.1, 1, 10])
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--num-prompts', type=int, default=10)
    parser.add_argument('--output-file', type=str, default='exp_2_results.json')
    args = parser.parse_args()

    start_time = time.time()
    
    print("Loading Wikipedia dataset...")
    chunks, embeddings_dict = load_wikipedia_dataset()
    
    # Initialize the core components
    embedder = EmbeddingGenerator()
    vector_store = VectorStore(reset=True)
    
    # Populate the vector store
    from core.data_models import EmbeddedChunk, Chunk
    embedded_chunks = [
        EmbeddedChunk(
            chunk=Chunk(id=cid, text=c.get('text', ''), source=c.get('source', 'unknown'), metadata=c), 
            embedding=emb
        ) for cid, c, emb in zip(embeddings_dict.keys(), chunks, embeddings_dict.values())
    ]
    vector_store.add(embedded_chunks)
    
    retriever = Retriever(embedder, vector_store)
    
    results = grid_search(retriever, chunks, embeddings_dict, **vars(args))
    
    output_path = project_root / 'results' / args.output_file
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}")

    plot_heatmap(results, project_root / 'results' / 'exp_2_heatmap.png')

    if results:
        best = results[0]
        print("\n--- Recommended Parameters ---")
        print(f"Alpha: {best['alpha']:.3f}, Penalty: {best['penalty']}")
        print(f"Achieved Recall: {best['overall_mean_aspect_recall']:.2f}%")
        print("----------------------------\n")

    print(f"Total time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()