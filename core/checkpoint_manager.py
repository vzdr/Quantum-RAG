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
                f.write(json.dumps(asdict(chunk)) + '\n')

    def _load_chunks(self) -> List:
        """Load chunks from JSONL."""
        from core.wikipedia.chunk_creator import Chunk

        chunks = []
        with open(self.checkpoint_dir / 'chunks.jsonl', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                chunks.append(Chunk(**data))
        return chunks

    def _save_embeddings(self, embedded_chunks: List):
        """Save embedded chunks as JSONL."""
        with open(self.checkpoint_dir / 'embeddings.jsonl', 'w', encoding='utf-8') as f:
            for embedded_chunk in embedded_chunks:
                data = {
                    "chunk": asdict(embedded_chunk.chunk),
                    "embedding": embedded_chunk.embedding.tolist()
                }
                f.write(json.dumps(data) + '\n')

    def _load_embeddings(self) -> List:
        """Load embedded chunks from JSONL."""
        from core.embedder import EmbeddedChunk
        from core.wikipedia.chunk_creator import Chunk

        embedded_chunks = []
        with open(self.checkpoint_dir / 'embeddings.jsonl', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                chunk = Chunk(**data['chunk'])
                embedding = np.array(data['embedding'])
                embedded_chunks.append(EmbeddedChunk(chunk=chunk, embedding=embedding))
        return embedded_chunks

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
