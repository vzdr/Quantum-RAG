"""
Text Chunker Module - Phase A, Step 2

Handles text chunking with various strategies:
- Fixed: Split by character count
- Sentence: Split at sentence boundaries
- Paragraph: Split at paragraph boundaries
"""
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from .document_loader import Document


@dataclass
class Chunk:
    """Represents a text chunk."""
    id: str
    text: str
    source: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """Return the length of the chunk text."""
        return len(self.text)

    @property
    def word_count(self) -> int:
        """Return the word count of the chunk."""
        return len(self.text.split())


class TextChunker:
    """
    Handles text chunking with various strategies.

    Strategies:
    - fixed: Split by character count
    - sentence: Split at sentence boundaries
    - paragraph: Split at paragraph boundaries
    """

    STRATEGIES = ['fixed', 'sentence', 'paragraph']

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50,
        strategy: str = "sentence"
    ):
        """
        Initialize the chunker.

        Args:
            chunk_size: Target size of each chunk in characters
            overlap: Number of characters to overlap between chunks
            strategy: Chunking strategy ('fixed', 'sentence', 'paragraph')
        """
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}. Choose from {self.STRATEGIES}")

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.strategy = strategy

    def chunk_document(self, document: Document) -> List[Chunk]:
        """
        Chunk a document based on the configured strategy.

        Args:
            document: Document to chunk

        Returns:
            List of Chunk objects
        """
        if self.strategy == "fixed":
            return self._fixed_chunking(document)
        elif self.strategy == "sentence":
            return self._sentence_chunking(document)
        elif self.strategy == "paragraph":
            return self._paragraph_chunking(document)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """
        Chunk multiple documents.

        Args:
            documents: List of documents to chunk

        Returns:
            List of all Chunk objects
        """
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        return all_chunks

    def _fixed_chunking(self, document: Document) -> List[Chunk]:
        """
        Simple fixed-size chunking.

        Args:
            document: Document to chunk

        Returns:
            List of Chunk objects
        """
        text = document.content
        chunks = []
        start = 0
        chunk_idx = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]

            # Skip empty chunks
            if chunk_text.strip():
                chunks.append(Chunk(
                    id=f"{document.source}_{chunk_idx}",
                    text=chunk_text,
                    source=document.source,
                    chunk_index=chunk_idx,
                    start_char=start,
                    end_char=end,
                    metadata={
                        'strategy': 'fixed',
                        'document_type': document.file_type,
                    }
                ))
                chunk_idx += 1

            # Move to next position with overlap
            if end >= len(text):
                break
            start = max(start + 1, end - self.overlap)

        return chunks

    def _sentence_chunking(self, document: Document) -> List[Chunk]:
        """
        Sentence-aware chunking.

        Splits text at sentence boundaries while respecting chunk size.

        Args:
            document: Document to chunk

        Returns:
            List of Chunk objects
        """
        # Split into sentences using regex
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, document.content)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk_sentences = []
        current_length = 0
        chunk_idx = 0
        current_start = 0
        char_position = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            # Check if adding this sentence would exceed chunk size
            if current_length + sentence_len + 1 > self.chunk_size and current_chunk_sentences:
                # Create chunk from accumulated sentences
                chunk_text = ' '.join(current_chunk_sentences)
                chunks.append(Chunk(
                    id=f"{document.source}_{chunk_idx}",
                    text=chunk_text,
                    source=document.source,
                    chunk_index=chunk_idx,
                    start_char=current_start,
                    end_char=current_start + len(chunk_text),
                    metadata={
                        'strategy': 'sentence',
                        'sentence_count': len(current_chunk_sentences),
                        'document_type': document.file_type,
                    }
                ))
                chunk_idx += 1

                # Handle overlap: keep last portion of sentences
                overlap_sentences = []
                overlap_len = 0
                for s in reversed(current_chunk_sentences):
                    if overlap_len + len(s) <= self.overlap:
                        overlap_sentences.insert(0, s)
                        overlap_len += len(s) + 1
                    else:
                        break

                current_chunk_sentences = overlap_sentences
                current_length = overlap_len
                current_start = char_position - overlap_len

            current_chunk_sentences.append(sentence)
            current_length += sentence_len + 1
            char_position += sentence_len + 1

        # Add remaining sentences as final chunk
        if current_chunk_sentences:
            chunk_text = ' '.join(current_chunk_sentences)
            chunks.append(Chunk(
                id=f"{document.source}_{chunk_idx}",
                text=chunk_text,
                source=document.source,
                chunk_index=chunk_idx,
                start_char=current_start,
                end_char=len(document.content),
                metadata={
                    'strategy': 'sentence',
                    'sentence_count': len(current_chunk_sentences),
                    'document_type': document.file_type,
                }
            ))

        return chunks

    def _paragraph_chunking(self, document: Document) -> List[Chunk]:
        """
        Paragraph-aware chunking.

        Splits text at paragraph boundaries while respecting chunk size.

        Args:
            document: Document to chunk

        Returns:
            List of Chunk objects
        """
        # Split into paragraphs
        paragraphs = re.split(r'\n\s*\n', document.content)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks = []
        current_chunk_paragraphs = []
        current_length = 0
        chunk_idx = 0
        current_start = 0
        char_position = 0

        for para in paragraphs:
            para_len = len(para)

            # Check if adding this paragraph would exceed chunk size
            if current_length + para_len + 2 > self.chunk_size and current_chunk_paragraphs:
                # Create chunk from accumulated paragraphs
                chunk_text = '\n\n'.join(current_chunk_paragraphs)
                chunks.append(Chunk(
                    id=f"{document.source}_{chunk_idx}",
                    text=chunk_text,
                    source=document.source,
                    chunk_index=chunk_idx,
                    start_char=current_start,
                    end_char=current_start + len(chunk_text),
                    metadata={
                        'strategy': 'paragraph',
                        'paragraph_count': len(current_chunk_paragraphs),
                        'document_type': document.file_type,
                    }
                ))
                chunk_idx += 1

                # Reset with overlap consideration
                current_chunk_paragraphs = []
                current_length = 0
                current_start = char_position

            current_chunk_paragraphs.append(para)
            current_length += para_len + 2
            char_position += para_len + 2

        # Add remaining paragraphs as final chunk
        if current_chunk_paragraphs:
            chunk_text = '\n\n'.join(current_chunk_paragraphs)
            chunks.append(Chunk(
                id=f"{document.source}_{chunk_idx}",
                text=chunk_text,
                source=document.source,
                chunk_index=chunk_idx,
                start_char=current_start,
                end_char=len(document.content),
                metadata={
                    'strategy': 'paragraph',
                    'paragraph_count': len(current_chunk_paragraphs),
                    'document_type': document.file_type,
                }
            ))

        return chunks

    def get_statistics(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """
        Calculate statistics for a list of chunks.

        Args:
            chunks: List of chunks

        Returns:
            Dictionary with statistics
        """
        if not chunks:
            return {
                'total_chunks': 0,
                'avg_length': 0,
                'min_length': 0,
                'max_length': 0,
                'total_chars': 0,
                'avg_words': 0,
            }

        lengths = [len(c.text) for c in chunks]
        word_counts = [c.word_count for c in chunks]

        return {
            'total_chunks': len(chunks),
            'avg_length': sum(lengths) / len(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'total_chars': sum(lengths),
            'avg_words': sum(word_counts) / len(word_counts),
            'sources': list(set(c.source for c in chunks)),
        }

    def preview_chunks(self, chunks: List[Chunk], max_preview: int = 100) -> List[Dict[str, Any]]:
        """
        Generate preview of chunks for display.

        Args:
            chunks: List of chunks
            max_preview: Maximum characters to show per chunk

        Returns:
            List of preview dictionaries
        """
        previews = []
        for chunk in chunks:
            preview_text = chunk.text[:max_preview]
            if len(chunk.text) > max_preview:
                preview_text += "..."

            previews.append({
                'id': chunk.id,
                'preview': preview_text,
                'length': len(chunk.text),
                'words': chunk.word_count,
                'source': chunk.source,
            })
        return previews
