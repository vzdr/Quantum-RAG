"""
Experiment 2: Parameter Sensitivity Analysis

Purpose: Identify optimal QUBO parameters (alpha and beta) and validate robustness.
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
    retriever: Retriever, chunks, embeddings_dict, alpha, beta, prompt_ids, k=5,
    redundancy_levels=range(6)
):
    """
    Evaluates a single (alpha, beta) parameter combination.
    """
    total_recalls = []
    
    for level in redundancy_levels:
        level_recalls = []
        for prompt_id in prompt_ids:
            query_text = ""
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
                beta=beta,
                penalty=10.0  # Keep penalty fixed
            )
            
            _, gold_aspects, _, _, _ = filter_chunks_by_prompt(chunks, prompt_id, level)
            
            recall, _ = compute_aspect_recall([r.chunk for r in results], gold_aspects)
            level_recalls.append(recall)

        if not level_recalls:
            total_recalls.append(0)
        else:
            total_recalls.extend(level_recalls)

    return {
        'alpha': alpha,
        'beta': beta,
        'overall_mean_aspect_recall': np.mean(total_recalls),
    }

def grid_search(retriever: Retriever, chunks, embeddings_dict, **kwargs):
    """Performs grid search over alpha and beta parameters."""
    alpha_range = np.arange(kwargs['alpha_min'], kwargs['alpha_max'], kwargs['alpha_step'])
    beta_range = np.arange(kwargs['beta_min'], kwargs['beta_max'], kwargs['beta_step'])
    num_prompts = kwargs['num_prompts']
    
    all_prompt_ids = list(set(c['prompt_id'] for c in chunks if c.get('chunk_type') == 'prompt'))
    prompt_ids = np.random.choice(all_prompt_ids, size=min(num_prompts, len(all_prompt_ids)), replace=False).tolist()
    
    print(f"Starting grid search with {len(alpha_range) * len(beta_range)} combinations...")
    
    results = []
    param_combinations = [(b, a) for b in beta_range for a in alpha_range]

    for beta, alpha in tqdm(param_combinations, desc="Grid Search"):
        result = evaluate_parameters(
            retriever, chunks, embeddings_dict, alpha, beta, prompt_ids, k=kwargs['k']
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
    betas = sorted(list(set(r['beta'] for r in results)))
    heatmap_data = np.full((len(betas), len(alphas)), np.nan)

    for r in results:
        if r['alpha'] in alphas and r['beta'] in betas:
            alpha_idx = alphas.index(r['alpha'])
            beta_idx = betas.index(r['beta'])
            heatmap_data[beta_idx, alpha_idx] = r['overall_mean_aspect_recall']

    plt.figure(figsize=(14, 8))
    sns.heatmap(
        heatmap_data, xticklabels=[f"{a:.2f}" for a in alphas], yticklabels=[f"{b:.2f}" for b in betas],
        annot=True, fmt='.1f', cmap='RdYlGn', vmin=0, vmax=100,
        cbar_kws={'label': 'Average Aspect Recall (%)'}
    )
    
    plt.xlabel('Alpha (α) - Diversity Weight', fontweight='bold')
    plt.ylabel('Beta (β) - Similarity Threshold', fontweight='bold')
    plt.title('QUBO Parameter Grid Search: Alpha vs. Beta', fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Heatmap saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='QUBO Parameter Grid Search')
    parser.add_argument('--alpha-min', type=float, default=0.0)
    parser.add_argument('--alpha-max', type=float, default=0.1)
    parser.add_argument('--alpha-step', type=float, default=0.01)
    parser.add_argument('--beta-min', type=float, default=0.0)
    parser.add_argument('--beta-max', type=float, default=1.0)
    parser.add_argument('--beta-step', type=float, default=0.1)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--num-prompts', type=int, default=10)
    parser.add_argument('--output-file', type=str, default='exp_2_results.json')
    args = parser.parse_args()

    start_time = time.time()
    
    print("Loading Wikipedia dataset...")
    chunks, embeddings_dict = load_wikipedia_dataset()
    
    embedder = EmbeddingGenerator()
    vector_store = VectorStore(reset=True)
    
    # Populate the vector store safely
    from core.data_models import EmbeddedChunk, Chunk
    embedded_chunks = []
    for chunk_data in chunks:
        chunk_id = chunk_data.get('chunk_id')
        if chunk_id in embeddings_dict:
            chunk_obj = Chunk(id=chunk_id, text=chunk_data.get('text', ''), source=chunk_data.get('article_title', 'unknown'), metadata=chunk_data)
            embedding = embeddings_dict[chunk_id]
            embedded_chunks.append(EmbeddedChunk(chunk=chunk_obj, embedding=embedding))
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
        print(f"Alpha: {best['alpha']:.3f}, Beta: {best['beta']:.2f}")
        print(f"Achieved Recall: {best['overall_mean_aspect_recall']:.2f}%")
        print("----------------------------\n")

    print(f"Total time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()