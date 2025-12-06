"""
Checkpoint Manager for granular stage-based checkpointing.
Manages checkpoint files for articles, chunks, embeddings, and upload progress.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Any
from dataclasses import asdict


class CheckpointManager:
    """
    Manages granular checkpoints at each pipeline stage.

    Stages:
    1. articles - Fetched Wikipedia articles
    2. chunks - Created chunks with redundancy
    3. embeddings - Computed embeddings
    4. upload - Upload progress to Chroma
    """

    def __init__(self, checkpoint_dir: str = './data/wikipedia/checkpoints'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.status_file = self.checkpoint_dir / 'stage_status.json'

        # Load existing status
        if self.status_file.exists():
            with open(self.status_file) as f:
                self.status = json.load(f)
        else:
            self.status = {
                'articles': False,
                'chunks': False,
                'embeddings': False,
                'upload_complete': False
            }

    def has_stage(self, stage_name: str) -> bool:
        """Check if stage is completed."""
        return self.status.get(stage_name, False)

    def save_stage(self, stage_name: str, data: Any):
        """Save stage data and mark as complete."""
        if stage_name == 'articles':
            self._save_articles(data)
        elif stage_name == 'chunks':
            self._save_chunks(data)
        elif stage_name == 'embeddings':
            self._save_embeddings(data)

        self.status[stage_name] = True
        self._save_status()

    def load_stage(self, stage_name: str) -> Any:
        """Load stage data."""
        if stage_name == 'articles':
            return self._load_articles()
        elif stage_name == 'chunks':
            return self._load_chunks()
        elif stage_name == 'embeddings':
            return self._load_embeddings()

    def _save_articles(self, articles: List):
        """Save articles as JSONL."""
        with open(self.checkpoint_dir / 'articles.jsonl', 'w', encoding='utf-8') as f:
            for article in articles:
                f.write(json.dumps(article.to_dict()) + '\n')

    def _load_articles(self) -> List:
        """Load articles from JSONL."""
        from core.wikipedia.fetcher import WikiArticle

        articles = []
        with open(self.checkpoint_dir / 'articles.jsonl', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                articles.append(WikiArticle.from_dict(data))
        return articles

    def _save_chunks(self, chunks: List):
        """Save chunks as JSONL."""
        with open(self.checkpoint_dir / 'chunks.jsonl', 'w', encoding='utf-8') as f:
            for chunk in chunks:
                # Convert Chunk to dict, preserving old format for compatibility
                chunk_data = {
                    'chunk_id': chunk.id,
                    'text': chunk.text,
                    'article_title': chunk.source,
                    'start_char': chunk.start_char,
                    'end_char': chunk.end_char,
                    **chunk.metadata  # Include all metadata fields
                }
                f.write(json.dumps(chunk_data) + '\n')

    def _load_chunks(self) -> List:
        """Load chunks from JSONL."""
        from core.chunker import Chunk

        chunks = []
        with open(self.checkpoint_dir / 'chunks.jsonl', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                data = json.loads(line)

                # Map old format to new Chunk format
                chunk_id = data.get('chunk_id', f'chunk_{idx}')
                chunk = Chunk(
                    id=chunk_id,
                    text=data.get('text', ''),
                    source=data.get('article_title', 'unknown'),
                    chunk_index=idx,
                    start_char=data.get('start_char', 0),
                    end_char=data.get('end_char', len(data.get('text', ''))),
                    metadata={
                        'chunk_type': data.get('chunk_type', ''),
                        'prompt_id': data.get('prompt_id', ''),
                        'article_title': data.get('article_title', ''),
                        'aspect_id': data.get('aspect_id', -1),
                        'aspect_name': data.get('aspect_name', ''),
                        'redundancy_index': data.get('redundancy_index', -1)
                    }
                )
                chunks.append(chunk)
        return chunks

    def _save_embeddings(self, embedded_chunks: List):
        """Save embedded chunks as NPZ (keyed by chunk_id)."""
        # Build dictionary mapping chunk_id -> embedding
        embeddings_dict = {}
        for ec in embedded_chunks:
            chunk_id = ec.chunk.id  # Use 'id' attribute from Chunk
            embeddings_dict[chunk_id] = ec.embedding

        # Save embeddings as compressed NPZ (keyed by chunk_id)
        np.savez_compressed(
            self.checkpoint_dir / 'embeddings.npz',
            **embeddings_dict
        )

    def _load_embeddings(self) -> List:
        """Load embedded chunks from NPZ format (or fallback to JSONL)."""
        from core.embedder import EmbeddedChunk
        from core.chunker import Chunk

        npz_path = self.checkpoint_dir / 'embeddings.npz'
        chunks_path = self.checkpoint_dir / 'chunks.jsonl'

        # Try NPZ format (stored by chunk_id as keys)
        if npz_path.exists() and chunks_path.exists():
            # Load NPZ data (keyed by chunk_id)
            npz_data = np.load(npz_path)

            # Load chunks and match with embeddings
            embedded_chunks = []
            with open(chunks_path, encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    chunk_data = json.loads(line)

                    # Map old format to new Chunk format
                    chunk_id = chunk_data.get('chunk_id', f'chunk_{idx}')
                    chunk = Chunk(
                        id=chunk_id,
                        text=chunk_data.get('text', ''),
                        source=chunk_data.get('article_title', 'unknown'),
                        chunk_index=idx,
                        start_char=chunk_data.get('start_char', 0),
                        end_char=chunk_data.get('end_char', len(chunk_data.get('text', ''))),
                        metadata={
                            'chunk_type': chunk_data.get('chunk_type', ''),
                            'prompt_id': chunk_data.get('prompt_id', ''),
                            'article_title': chunk_data.get('article_title', ''),
                            'aspect_id': chunk_data.get('aspect_id', -1),
                            'aspect_name': chunk_data.get('aspect_name', ''),
                            'redundancy_index': chunk_data.get('redundancy_index', -1)
                        }
                    )

                    # Get embedding for this chunk_id
                    if chunk_id in npz_data:
                        embedding = npz_data[chunk_id]
                        embedded_chunks.append(EmbeddedChunk(chunk=chunk, embedding=embedding))
                    else:
                        # Fallback: try array format (embeddings stored as array)
                        if 'embeddings' in npz_data:
                            # New format with separate metadata
                            embedding = npz_data['embeddings'][idx]
                            embedded_chunks.append(EmbeddedChunk(chunk=chunk, embedding=embedding))

            return embedded_chunks

        # Fallback to JSONL format (legacy)
        jsonl_path = self.checkpoint_dir / 'embeddings.jsonl'
        if jsonl_path.exists():
            embedded_chunks = []
            with open(jsonl_path, encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    chunk = Chunk(**data['chunk'])
                    embedding = np.array(data['embedding'])
                    embedded_chunks.append(EmbeddedChunk(chunk=chunk, embedding=embedding))
            return embedded_chunks

        raise FileNotFoundError(f"No embeddings file found at {npz_path} or {jsonl_path}")

    def get_uploaded_ids(self) -> Set[str]:
        """Get list of already uploaded chunk IDs."""
        uploaded_file = self.checkpoint_dir / 'uploaded.json'
        if uploaded_file.exists():
            with open(uploaded_file, encoding='utf-8') as f:
                return set(json.load(f))
        return set()

    def mark_uploaded(self, chunk_ids: List[str]):
        """Mark chunk IDs as uploaded."""
        uploaded = self.get_uploaded_ids()
        uploaded.update(chunk_ids)

        with open(self.checkpoint_dir / 'uploaded.json', 'w', encoding='utf-8') as f:
            json.dump(list(uploaded), f)

    def mark_upload_complete(self):
        """Mark upload stage as complete."""
        self.status['upload_complete'] = True
        self._save_status()

    def _save_status(self):
        """Save completion status."""
        with open(self.status_file, 'w', encoding='utf-8') as f:
            json.dump(self.status, f, indent=2)

    def clear(self):
        """Clear all checkpoints."""
        for f in self.checkpoint_dir.glob('*'):
            if f.is_file():
                f.unlink()

        self.status = {
            'articles': False,
            'chunks': False,
            'embeddings': False,
            'upload_complete': False
        }
        self._save_status()
        print("Checkpoints cleared")
