"""
Experiment 2: Grid Search for Optimal Alpha and Beta in QUBO Retrieval.
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

# Setup paths and imports
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from core.retrieval import Retriever
from core.embedding import EmbeddingGenerator
from core.storage import VectorStore
from core.data_models import EmbeddedChunk, Chunk
from core.utils import load_wikipedia_dataset, filter_chunks_by_prompt, compute_aspect_recall

def evaluate_parameters(retriever, chunks, embeddings_dict, alpha, beta, prompt_ids, k, redundancy_levels):
    """
    Evaluates a single (alpha, beta) pair by averaging the mean recall across all redundancy levels.
    """
    level_average_recalls = []

    for level in redundancy_levels:
        prompt_recalls_for_level = []
        for prompt_id in prompt_ids:
            # Find the query text for the current prompt
            query_text = next((c['text'] for c in chunks if c.get('chunk_type') == 'prompt' and c.get('prompt_id') == prompt_id), None)
            if not query_text:
                continue

            # Get candidates and ground truth for this specific prompt and level
            candidates, gold_aspects, _, _, _ = filter_chunks_by_prompt(chunks, prompt_id, level)
            if len(candidates) < k or not gold_aspects:
                continue
            
            # The retriever needs the full candidate info, including embeddings
            candidate_results_with_embeddings = [
                {
                    'id': c['chunk_id'],
                    'text': c['text'],
                    'embedding': embeddings_dict.get(c['chunk_id']),
                    'score': 0,  # Score is recalculated inside the strategy
                    'metadata': c
                } for c in candidates if c['chunk_id'] in embeddings_dict
            ]

            # Run retrieval
            try:
                results = retriever.retrieve(
                    query=query_text,
                    k=k,
                    strategy='qubo',
                    candidates=candidate_results_with_embeddings,
                    alpha=alpha,
                    beta=beta
                )
                
                # Calculate recall and add to list for this level
                recall, _ = compute_aspect_recall([r.chunk for r in results], gold_aspects)
                prompt_recalls_for_level.append(recall)
            except Exception:
                # If a solver or other error occurs, count it as 0 recall for this prompt
                prompt_recalls_for_level.append(0)

        # Average the recalls for the current redundancy level
        if prompt_recalls_for_level:
            avg_recall_for_level = np.mean(prompt_recalls_for_level)
            level_average_recalls.append(avg_recall_for_level)

    # The final score is the average of the mean recalls from each level
    if not level_average_recalls:
        return None
        
    final_score = np.mean(level_average_recalls)
    return {'alpha': alpha, 'beta': beta, 'score': final_score}

def grid_search(retriever, chunks, embeddings_dict, args):
    """Performs grid search over alpha and beta."""
    alpha_range = np.arange(args.alpha_min, args.alpha_max, args.alpha_step)
    beta_range = np.arange(args.beta_min, args.beta_max, args.beta_step)
    
    all_prompt_ids = list(set(c['prompt_id'] for c in chunks if c.get('chunk_type') == 'prompt'))
    prompt_ids = np.random.choice(all_prompt_ids, size=min(args.num_prompts, len(all_prompt_ids)), replace=False).tolist()
    
    print(f"Starting grid search over {len(alpha_range) * len(beta_range)} combinations on {len(prompt_ids)} prompts...")
    
    results = []
    param_combinations = [(a, b) for a in alpha_range for b in beta_range]

    for alpha, beta in tqdm(param_combinations, desc="Grid Search"):
        result = evaluate_parameters(
            retriever, chunks, embeddings_dict, alpha, beta, prompt_ids, args.k, args.redundancy_levels
        )
        if result:
            results.append(result)
            
    results.sort(key=lambda x: x['score'], reverse=True)
    return results

def plot_heatmap(results, output_path):
    """Creates and saves a heatmap of the grid search results."""
    if not results:
        print("No results to plot.")
        return

    alphas = sorted(list(set(r['alpha'] for r in results)))
    betas = sorted(list(set(r['beta'] for r in results)))
    heatmap_data = np.full((len(betas), len(alphas)), np.nan)

    for r in results:
        alpha_idx = alphas.index(r['alpha'])
        beta_idx = betas.index(r['beta'])
        heatmap_data[beta_idx, alpha_idx] = r['score']

    plt.figure(figsize=(14, 8))
    sns.heatmap(
        heatmap_data, xticklabels=[f"{a:.2f}" for a in alphas], yticklabels=[f"{b:.2f}" for b in betas],
        annot=True, fmt='.1f', cmap='viridis', cbar_kws={'label': 'Mean Aspect Recall (%)'}
    )
    
    plt.xlabel('Alpha (α) - Diversity Weight')
    plt.ylabel('Beta (β) - Similarity Threshold')
    plt.title('QUBO Parameter Tuning: Mean Aspect Recall')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"\nHeatmap saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Grid search for QUBO alpha and beta parameters.')
    parser.add_argument('--num-prompts', type=int, default=10, help='Number of prompts to test.')
    parser.add_argument('--k', type=int, default=5, help='Number of chunks to retrieve.')
    parser.add_argument('--alpha-min', type=float, default=0.0, help='Min alpha value.')
    parser.add_argument('--alpha-max', type=float, default=0.2, help='Max alpha value.')
    parser.add_argument('--alpha-step', type=float, default=0.02, help='Step for alpha.')
    parser.add_argument('--beta-min', type=float, default=0.0, help='Min beta value.')
    parser.add_argument('--beta-max', type=float, default=0.8, help='Max beta value.')
    parser.add_argument('--beta-step', type=float, default=0.1, help='Step for beta.')
    parser.add_argument('--redundancy-levels', type=int, nargs='+', default=list(range(6)), help='Redundancy levels to test.')
    args = parser.parse_args()

    start_time = time.time()
    
    print("Loading dataset...")
    chunks, embeddings_dict = load_wikipedia_dataset()
    
    print("Initializing RAG components...")
    embedder = EmbeddingGenerator()
    vector_store = VectorStore(reset=True)
    
    embedded_chunks = [
        EmbeddedChunk(
            chunk=Chunk(id=cid, text=c.get('text', ''), source=c.get('article_title', 'unknown'), metadata=c), 
            embedding=emb
        ) for cid, c, emb in zip(embeddings_dict.keys(), chunks, embeddings_dict.values()) if cid in embeddings_dict and c.get('chunk_id') == cid
    ]
    vector_store.add(embedded_chunks)
    
    retriever = Retriever(embedder, vector_store)
    
    results = grid_search(retriever, chunks, embeddings_dict, args)
    
    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / 'exp_2_parameter_tuning.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    plot_heatmap(results, results_dir / 'exp_2_heatmap.png')

    if results:
        best = results[0]
        print("\n--- Best Parameters ---")
        print(f"Alpha: {best['alpha']:.3f}, Beta: {best['beta']:.2f}")
        print(f"Score (Mean Aspect Recall): {best['score']:.2f}%")
        print("---------------------\\n")

    print(f"Total execution time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
