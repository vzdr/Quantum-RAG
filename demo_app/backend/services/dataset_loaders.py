"""
Dataset loaders for various formats.
Handles Wikipedia JSONL+NPZ format from competition notebook.
"""
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any


def load_wikipedia_dataset(data_dir: Path) -> Tuple[List[Dict], Dict[str, np.ndarray]]:
    """
    Load Wikipedia dataset from notebook format.

    Args:
        data_dir: Path to data directory containing checkpoints folder

    Returns:
        chunks: List of chunk dictionaries
        embeddings: Dictionary mapping chunk_id -> embedding vector
    """
    # Load chunks from JSONL
    chunks_file = data_dir / 'checkpoints' / 'chunks.jsonl'
    if not chunks_file.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_file}")

    chunks = []
    with open(chunks_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                chunks.append(json.loads(line))

    # Load embeddings from NPZ
    embeddings_file = data_dir / 'checkpoints' / 'embeddings.npz'
    if not embeddings_file.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")

    embeddings_npz = np.load(embeddings_file)
    embeddings = {key: embeddings_npz[key] for key in embeddings_npz.keys()}

    print(f"Loaded {len(chunks)} chunks and {len(embeddings)} embeddings from Wikipedia dataset")

    return chunks, embeddings


def convert_wikipedia_to_chroma_format(
    chunks: List[Dict],
    embeddings: Dict[str, np.ndarray],
    redundancy_level: int = 4
) -> List[Dict[str, Any]]:
    """
    Convert Wikipedia dataset to ChromaDB-compatible format.
    Keeps gold_base, noise, and up to redundancy_level copies of each gold aspect.

    Args:
        chunks: Wikipedia chunks from JSONL
        embeddings: Embeddings dict
        redundancy_level: Number of redundant copies to keep per aspect (default: 4)

    Returns:
        List of chunks in ChromaDB format with id, text, embedding, metadata
    """
    chroma_chunks = []

    filtered_count = {'prompt': 0, 'gold_redundant_kept': 0, 'gold_redundant_filtered': 0, 'kept': 0}

    # Track how many redundant chunks we've kept per (article, aspect) combination
    redundant_counts = {}

    for chunk in chunks:
        chunk_type = chunk.get('chunk_type', '')

        # Skip prompts
        if chunk_type == 'prompt':
            filtered_count['prompt'] += 1
            continue

        # For redundant chunks, keep only redundancy_level copies per aspect
        if chunk_type == 'gold_redundant':
            article_title = chunk.get('article_title', 'unknown')
            aspect_name = chunk.get('aspect_name', 'general')
            key = (article_title, aspect_name)

            # Count how many we've kept so far
            count = redundant_counts.get(key, 0)

            if count < redundancy_level:
                # Keep this redundant chunk
                redundant_counts[key] = count + 1
                filtered_count['gold_redundant_kept'] += 1
            else:
                # Skip this redundant chunk (already have enough)
                filtered_count['gold_redundant_filtered'] += 1
                continue

        chunk_id = chunk['chunk_id']
        if chunk_id not in embeddings:
            continue

        # Extract metadata for cluster labeling
        aspect_name = chunk.get('aspect_name', 'general')
        aspect_id = chunk.get('aspect_id', -1)
        article_title = chunk.get('article_title', 'wikipedia')
        prompt_id = chunk.get('prompt_id', 'unknown')

        # Create source identifier (used for cluster counting)
        # Format: article_aspect (e.g., "Lupus_symptoms", "Diabetes_treatment")
        source_name = f"{article_title}_{aspect_name}"

        chroma_chunks.append({
            'id': chunk_id,
            'text': chunk['text'],
            'embedding': embeddings[chunk_id],
            'metadata': {
                'source': source_name,
                'aspect_name': aspect_name,
                'aspect_id': aspect_id,
                'article_title': article_title,
                'chunk_type': chunk_type,
                'prompt_id': prompt_id,
                'chunk_index': 0,  # Not used in Wikipedia format
                'start_char': 0,
                'end_char': len(chunk['text'])
            }
        })
        filtered_count['kept'] += 1

    print(f"Wikipedia filtering (redundancy_level={redundancy_level}): "
          f"Kept {filtered_count['kept']} total chunks "
          f"({filtered_count['gold_redundant_kept']} redundant + others), "
          f"Filtered {filtered_count['prompt']} prompts + "
          f"{filtered_count['gold_redundant_filtered']} excess redundant chunks")

    return chroma_chunks
