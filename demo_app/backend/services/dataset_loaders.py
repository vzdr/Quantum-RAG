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
    embeddings: Dict[str, np.ndarray]
) -> List[Dict[str, Any]]:
    """
    Convert Wikipedia dataset to ChromaDB-compatible format.
    Filters out redundant chunks and prompts - keeps only gold_base and noise.

    Args:
        chunks: Wikipedia chunks from JSONL
        embeddings: Embeddings dict

    Returns:
        List of chunks in ChromaDB format with id, text, embedding, metadata
    """
    chroma_chunks = []

    filtered_count = {'prompt': 0, 'gold_redundant': 0, 'kept': 0}

    for chunk in chunks:
        chunk_type = chunk.get('chunk_type', '')

        # Skip prompts and redundant chunks (keep only base + noise)
        if chunk_type == 'prompt':
            filtered_count['prompt'] += 1
            continue

        if chunk_type == 'gold_redundant':
            filtered_count['gold_redundant'] += 1
            continue

        chunk_id = chunk['chunk_id']
        if chunk_id not in embeddings:
            continue

        # Extract aspect name for cluster labeling
        aspect_name = chunk.get('aspect_name', 'general')
        article_title = chunk.get('article_title', 'wikipedia')

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
                'article_title': article_title,
                'chunk_type': chunk_type,
                'chunk_index': 0,  # Not used in Wikipedia format
                'start_char': 0,
                'end_char': len(chunk['text'])
            }
        })
        filtered_count['kept'] += 1

    print(f"Wikipedia filtering: Kept {filtered_count['kept']}, "
          f"Filtered out {filtered_count['prompt']} prompts + "
          f"{filtered_count['gold_redundant']} redundant chunks")

    return chroma_chunks
