"""
Shared utilities for calculations and data processing.
"""
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

def compute_cosine_similarities(query_emb: np.ndarray, candidate_embs: np.ndarray):
    """Computes cosine similarities between a query and candidate embeddings."""
    query_norm = query_emb / np.linalg.norm(query_emb)
    cand_norms = candidate_embs / np.linalg.norm(candidate_embs, axis=1, keepdims=True)
    return np.dot(cand_norms, query_norm)

def compute_pairwise_similarities(embeddings: np.ndarray) -> np.ndarray:
    """Computes the pairwise cosine similarity matrix for a set of embeddings."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / (norms + 1e-8)
    similarity_matrix = normalized_embeddings @ normalized_embeddings.T
    np.fill_diagonal(similarity_matrix, 1.0)
    return similarity_matrix

def load_wikipedia_dataset(data_dir: str = './data/wikipedia') -> Tuple[List[Dict], Dict[str, np.ndarray]]:
    """
    Load Wikipedia dataset with chunks and embeddings.
    """
    data_path = Path(data_dir)
    chunks = []
    chunks_file = data_path / 'checkpoints' / 'chunks.jsonl'
    with open(chunks_file, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))

    embeddings_file = data_path / 'checkpoints' / 'embeddings.npz'
    embeddings_npz = np.load(embeddings_file)
    embeddings = {key: embeddings_npz[key] for key in embeddings_npz.keys()}
    return chunks, embeddings

def filter_chunks_by_prompt(chunks: List[Dict],
                            prompt_id: str,
                            redundancy_level: int = 0) -> Tuple[List[Dict], set, List[Dict], List[Dict], int]:
    """
    Filter chunks for a specific prompt at a given redundancy level.
    """
    candidate_chunks = []
    gold_base_aspects = set()
    gold_chunks = []
    noise_chunks = []

    for chunk in chunks:
        if chunk.get('prompt_id') != prompt_id:
            continue

        chunk_type = chunk.get('chunk_type', '')
        aspect_id = chunk.get('aspect_id', -1)

        if chunk_type == 'gold_base':
            candidate_chunks.append(chunk)
            gold_chunks.append(chunk)
            if aspect_id >= 0:
                gold_base_aspects.add(aspect_id)
        elif chunk_type == 'gold_redundant':
            redundancy_idx = chunk.get('redundancy_index', -1)
            if redundancy_idx <= redundancy_level:
                candidate_chunks.append(chunk)
                gold_chunks.append(chunk)
        elif chunk_type == 'noise':
            candidate_chunks.append(chunk)
            noise_chunks.append(chunk)
    return candidate_chunks, gold_base_aspects, gold_chunks, noise_chunks, len(gold_base_aspects)

def get_prompt_embedding(chunks: List[Dict],
                         embeddings: Dict,
                         prompt_id: int) -> Optional[np.ndarray]:
    """Get the embedding for a specific prompt."""
    prompt_chunks = [c for c in chunks
                    if c.get('chunk_type') == 'prompt'
                    and c.get('prompt_id') == prompt_id]
    if not prompt_chunks:
        return None
    chunk_id = prompt_chunks[0]['chunk_id']
    return embeddings.get(chunk_id)

def compute_aspect_recall(retrieved_chunks: List,
                         gold_aspects: set) -> Tuple[float, int]:
    """
    Compute aspect recall: how many distinct gold aspects were retrieved?
    """
    retrieved_aspects = set()
    for chunk in retrieved_chunks:
        # This now handles both dicts and Chunk objects
        if hasattr(chunk, 'metadata'):
            aspect_id = chunk.metadata.get('aspect_id', -1)
        else:
            aspect_id = chunk.get('aspect_id', -1)
            
        if aspect_id >= 0:
            retrieved_aspects.add(aspect_id)
            
    num_gold = len(gold_aspects)
    if num_gold == 0:
        return 0.0, 0
        
    num_retrieved = len(retrieved_aspects & gold_aspects)
    recall = 100.0 * num_retrieved / num_gold
    return recall, num_retrieved

def print_retrieval_results(retrieved_chunks: List[Dict],
                           gold_aspects: set,
                           method_name: str = "Retrieval"):
    """Print retrieval results in a readable format."""
    print(f"\n{'='*80}")
    print(f"{method_name.upper()} RESULTS")
    print(f"{ '='*80}")
    for i, chunk in enumerate(retrieved_chunks, 1):
        chunk_type = chunk.get('chunk_type', 'unknown')
        aspect_id = chunk.get('aspect_id', -1)
        aspect_name = chunk.get('aspect_name', 'N/A')
        is_gold = aspect_id in gold_aspects if aspect_id >= 0 else False
        gold_marker = '⭐' if is_gold else ''
        print(f"\n[{i}] Aspect {aspect_id}: {aspect_name} | {chunk_type} {gold_marker}")
        text = chunk.get('text', '')
        preview = text[:120] + '...' if len(text) > 120 else text
        print(f"    \"{preview}\"")
    recall, num_retrieved = compute_aspect_recall(retrieved_chunks, gold_aspects)
    print(f"\n{'-'*80}")
    print(f"Aspect Recall: {recall:.0f}% ({num_retrieved}/{len(gold_aspects)} gold aspects retrieved)")
    print(f"Total Retrieved: {len(retrieved_chunks)} chunks")
    print(f"{ '='*80}\n")

def extract_disease_from_filename(filename: str) -> str:
    """Extract disease/topic name from filename (e.g., 'Lupus_0.txt' -> 'Lupus')."""
    name = Path(filename).stem if '.' in filename else filename
    return name.split('_')[0]

def compute_intra_list_similarity(results: List[Dict]) -> float:
    """Compute average pairwise similarity within a result list."""
    if len(results) < 2:
        return 0.0
    embeddings = np.array([r['embedding'] for r in results])
    pairwise_sim = compute_pairwise_similarities(embeddings)
    mask = np.triu(np.ones((len(results), len(results))), k=1).astype(bool)
    return float(np.mean(pairwise_sim[mask]))

def compute_cluster_coverage_from_filenames(results: List[Dict], total_clusters: int) -> Dict:
    """Compute cluster coverage based on source filenames."""
    if not results:
        return {'coverage_count': 0, 'coverage_ratio': 0.0}
    clusters = {extract_disease_from_filename(r.get('source', '')) for r in results if r.get('source')}
    return {'coverage_count': len(clusters), 'coverage_ratio': len(clusters) / total_clusters if total_clusters > 0 else 0.0}

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot_product / (norm1 * norm2))

def retrieve_topk(query_embedding: np.ndarray, candidate_chunks: List[Dict], candidate_embeddings: Dict, k: int = 5) -> List[Dict]:
    """Standard Top-K retrieval: select k chunks with highest similarity."""
    scored_chunks = []
    for chunk in candidate_chunks:
        chunk_id = chunk['chunk_id']
        chunk_emb = candidate_embeddings.get(chunk_id)
        if chunk_emb is None:
            continue
        similarity = cosine_similarity(query_embedding, chunk_emb)
        scored_chunks.append({'chunk': chunk, 'score': similarity})
    scored_chunks.sort(key=lambda x: x['score'], reverse=True)
    return [item['chunk'] for item in scored_chunks[:k]]

def print_comparison_table(topk_results: Dict, qubo_results: Dict, gold_aspects: set, embeddings: Dict):
    """Print side-by-side comparison of Top-K vs QUBO."""
    topk_chunks = topk_results['chunks']
    qubo_chunks = qubo_results['chunks']
    topk_recall, topk_count = compute_aspect_recall(topk_chunks, gold_aspects)
    qubo_recall, qubo_count = compute_aspect_recall(qubo_chunks, gold_aspects)

    # Build results with embeddings for intra-list similarity
    topk_with_emb = [{'embedding': embeddings[c['chunk_id']]} for c in topk_chunks if c['chunk_id'] in embeddings]
    qubo_with_emb = [{'embedding': embeddings[c['chunk_id']]} for c in qubo_chunks if c['chunk_id'] in embeddings]
    topk_diversity = compute_intra_list_similarity(topk_with_emb)
    qubo_diversity = compute_intra_list_similarity(qubo_with_emb)

    print("\n" + "="*80)
    print("COMPARISON: TOP-K vs QUBO-RAG")
    print("="*80)
    print(f"{'Metric':<40} {'Top-K':>15} {'QUBO-RAG':>15}")
    print("-"*80)
    print(f"{'Aspect Recall (%)':<40} {topk_recall:>15.1f} {qubo_recall:>15.1f}")
    print(f"{'Aspects Retrieved (out of ' + str(len(gold_aspects)) + ')':<40} {topk_count:>15} {qubo_count:>15}")
    print(f"{'Intra-List Similarity (lower=diverse)':<40} {topk_diversity:>15.3f} {qubo_diversity:>15.3f}")
    print(f"{'Avg Relevance Score':<40} {topk_results.get('avg_relevance', 0):>15.3f} {qubo_results.get('avg_relevance', 0):>15.3f}")
    print(f"{'Chunks Retrieved':<40} {len(topk_chunks):>15} {len(qubo_chunks):>15}")
    if 'solve_time' in qubo_results:
        print(f"{'QUBO Solve Time (s)':<40} {'-':>15} {qubo_results['solve_time']:>15.3f}")
    print("="*80)
    if topk_recall > 0:
        improvement = ((qubo_recall - topk_recall) / topk_recall) * 100
        print(f"\n📊 QUBO improves aspect recall by {improvement:+.1f}%")
    diversity_improvement = ((topk_diversity - qubo_diversity) / topk_diversity) * 100 if topk_diversity > 0 else 0
    print(f"📊 QUBO reduces redundancy by {diversity_improvement:+.1f}%")
    print()