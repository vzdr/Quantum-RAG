"""
Chunk Creator for generating gold chunks with redundancy and noise chunks.
Processes articles to create base facts and redundant overlapping chunks.
"""

import random
from typing import List, Tuple
from dataclasses import dataclass
from uuid import uuid4


# Sections to skip (common non-content sections)
SKIP_SECTIONS = {
    'see also', 'references', 'external links', 'notes',
    'further reading', 'bibliography', 'gallery', 'sources',
    'citations', 'footnotes', 'works cited', 'external links',
    'notes and references', 'references and notes'
}


from core.chunker import Chunk


class ChunkCreator:
    """
    Creates chunks with redundancy from Wikipedia articles.

    Strategy:
    - Extract first 5 valid content sections
    - For each section: create 1 base chunk + N redundant chunks
    - Redundant chunks = base fact + additional context
    - Add 25 noise chunks from noise pool per prompt
    """

    def __init__(self, max_redundancy: int = 5):
        """
        Args:
            max_redundancy: Maximum redundancy level (creates max_redundancy + 1 chunks per aspect)
        """
        self.max_redundancy = max_redundancy
        self.noise_pool = []

    def build_noise_pool(self, noise_articles: List) -> int:
        """
        Build global noise pool from articles.

        Args:
            noise_articles: List of WikiArticle objects to use for noise

        Returns:
            Number of paragraphs in noise pool
        """
        self.noise_pool = []

        for article in noise_articles:
            for section in article.sections:
                # Skip common sections
                if section.title.lower() in SKIP_SECTIONS:
                    continue

                # Add all valid paragraphs to noise pool
                for paragraph in section.paragraphs:
                    if self._is_valid_paragraph(paragraph):
                        self.noise_pool.append(paragraph)

        return len(self.noise_pool)

    def _is_valid_paragraph(self, text: str) -> bool:
        """Check if paragraph meets quality criteria."""
        text_len = len(text.strip())
        return (
            100 <= text_len <= 3000 and
            not text.startswith('[') and  # Skip citation markers
            not text.startswith('!')      # Skip info boxes
        )

    def _get_valid_sections(self, article) -> List:
        """
        Get first 5 valid content sections from article.

        Args:
            article: WikiArticle object

        Returns:
            List of valid WikiSection objects (max 5)
        """
        valid_sections = []

        for section in article.sections:
            # Skip common non-content sections
            if section.title.lower() in SKIP_SECTIONS:
                continue

            # Check section has enough valid paragraphs
            valid_paragraphs = [p for p in section.paragraphs if self._is_valid_paragraph(p)]

            if len(valid_paragraphs) >= 2:  # Need at least 2 for redundancy
                # Update section with only valid paragraphs
                section.paragraphs = valid_paragraphs
                valid_sections.append(section)

            # Stop once we have 5 sections
            if len(valid_sections) >= 5:
                break

        return valid_sections

    def create_chunks_for_article(self, article) -> Tuple[Chunk, List[Chunk], List[Chunk]]:
        """
        Create all chunks for a single article.

        Args:
            article: WikiArticle object

        Returns:
            (prompt_chunk, gold_chunks, noise_chunks)
            Returns None if article doesn't meet requirements
        """
        # Get valid sections
        valid_sections = self._get_valid_sections(article)

        if len(valid_sections) < 5:
            return None  # Not enough valid sections

        # Generate prompt ID for this article
        prompt_id = str(uuid4())

        # Create prompt chunk
        prompt_text = (
            f"Provide a comprehensive overview of {article.title}, "
            f"covering key aspects such as {', '.join([s.title for s in valid_sections[:3]])}."
        )

        prompt_chunk = Chunk(
            id=str(uuid4()),
            text=prompt_text,
            source=article.title,
            chunk_index=0,
            start_char=0,
            end_char=len(prompt_text),
            metadata={
                'chunk_type': 'prompt',
                'prompt_id': prompt_id,
                'aspect_id': -1,
                'aspect_name': 'prompt',
                'redundancy_index': -1
            }
        )

        # Create gold chunks (base + redundant)
        gold_chunks = []

        for aspect_id, section in enumerate(valid_sections):
            paragraphs = section.paragraphs

            # Base chunk (just first paragraph)
            base_chunk = Chunk(
                id=str(uuid4()),
                text=paragraphs[0],
                source=article.title,
                chunk_index=0,
                start_char=0,
                end_char=len(paragraphs[0]),
                metadata={
                    'chunk_type': 'gold_base',
                    'prompt_id': prompt_id,
                    'aspect_id': aspect_id,
                    'aspect_name': section.title,
                    'redundancy_index': 0
                }
            )
            gold_chunks.append(base_chunk)

            # Redundant chunks (base + context)
            for redundancy_idx in range(1, self.max_redundancy + 1):
                # Select context paragraph (wrap around if needed)
                context_idx = redundancy_idx % len(paragraphs)
                if context_idx == 0:
                    context_idx = 1  # Don't use base as context

                redundant_text = paragraphs[0] + " " + paragraphs[context_idx]

                redundant_chunk = Chunk(
                    id=str(uuid4()),
                    text=redundant_text,
                    source=article.title,
                    chunk_index=redundancy_idx,
                    start_char=0,
                    end_char=len(redundant_text),
                    metadata={
                        'chunk_type': 'gold_redundant',
                        'prompt_id': prompt_id,
                        'aspect_id': aspect_id,
                        'aspect_name': section.title,
                        'redundancy_index': redundancy_idx
                    }
                )
                gold_chunks.append(redundant_chunk)

        # Create noise chunks
        noise_chunks = self._sample_noise(25, prompt_id, article.title)

        return prompt_chunk, gold_chunks, noise_chunks

    def _sample_noise(self, count: int, prompt_id: str, article_title: str) -> List[Chunk]:
        """
        Sample noise paragraphs from noise pool.

        Args:
            count: Number of noise paragraphs to sample
            prompt_id: Prompt ID for metadata
            article_title: Article title for metadata

        Returns:
            List of noise Chunk objects
        """
        if len(self.noise_pool) < count:
            raise ValueError(
                f"Noise pool has only {len(self.noise_pool)} paragraphs, "
                f"but {count} requested. Build larger noise pool."
            )

        sampled_paragraphs = random.sample(self.noise_pool, count)

        noise_chunks = []
        for i, paragraph in enumerate(sampled_paragraphs):
            noise_chunk = Chunk(
                id=str(uuid4()),
                text=paragraph,
                source='noise',
                chunk_index=i,
                start_char=0,
                end_char=len(paragraph),
                metadata={
                    'chunk_type': 'noise',
                    'prompt_id': prompt_id,
                    'article_title': article_title,
                    'aspect_id': -1,
                    'aspect_name': 'noise',
                    'redundancy_index': -1
                }
            )
            noise_chunks.append(noise_chunk)

        return noise_chunks
