"""
Wikipedia Fetcher with caching and reliability features - FIXED VERSION.
Handles article fetching with retry logic, validation, and local caching.
NOW PROPERLY EXTRACTS REAL SECTION NAMES.
"""

import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
import wikipedia
from dataclasses import dataclass, asdict


@dataclass
class WikiSection:
    """Represents a section from a Wikipedia article."""
    title: str
    paragraphs: List[str]
    level: int

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'WikiSection':
        return cls(
            title=data['title'],
            paragraphs=data['paragraphs'],
            level=data['level']
        )


@dataclass
class WikiArticle:
    """Represents a complete Wikipedia article."""
    title: str
    summary: str
    sections: List[WikiSection]
    url: str

    def to_dict(self) -> Dict:
        return {
            'title': self.title,
            'summary': self.summary,
            'sections': [s.to_dict() for s in self.sections],
            'url': self.url
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'WikiArticle':
        return cls(
            title=data['title'],
            summary=data['summary'],
            sections=[WikiSection.from_dict(s) for s in data['sections']],
            url=data['url']
        )


class WikipediaFetcher:
    """
    Fetches Wikipedia articles with caching and error handling.

    Features:
    - Local caching to avoid re-fetching
    - Retry logic with exponential backoff
    - Article validation (minimum section requirements)
    - Paragraph extraction and filtering
    - PROPERLY EXTRACTS REAL SECTION NAMES
    """

    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        # Wikipedia API settings
        wikipedia.set_lang("en")
        wikipedia.set_rate_limiting(True)

        # Quality filters
        self.min_paragraph_chars = 100
        self.max_paragraph_chars = 1500
        self.min_sections_required = 5

        # Sections to ignore
        self.ignore_sections = {
            'see also', 'references', 'external links', 'notes',
            'further reading', 'bibliography', 'gallery', 'sources',
            'citations', 'footnotes', 'works cited', 'notes and references',
            'references and notes', 'external links'
        }

    def _get_cache_path(self, title: str) -> Path:
        """Generate cache file path for article."""
        # Use hash to handle special characters in titles
        title_hash = hashlib.md5(title.encode()).hexdigest()
        return self.cache_dir / f"{title_hash}.json"

    def _load_from_cache(self, title: str) -> Optional[WikiArticle]:
        """Load article from cache if it exists."""
        cache_path = self._get_cache_path(title)
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    sections = [
                        WikiSection(
                            title=s['title'],
                            paragraphs=s['paragraphs'],
                            level=s['level']
                        )
                        for s in data['sections']
                    ]
                    return WikiArticle(
                        title=data['title'],
                        summary=data['summary'],
                        sections=sections,
                        url=data['url']
                    )
            except Exception:
                # If cache is corrupted, ignore and re-fetch
                pass
        return None

    def _save_to_cache(self, article: WikiArticle):
        """Save article to cache."""
        cache_path = self._get_cache_path(article.title)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(article.to_dict(), f, indent=2, ensure_ascii=False)

    def _filter_paragraph(self, paragraph: str) -> bool:
        """Check if paragraph meets quality criteria."""
        para_len = len(paragraph.strip())
        return (
            self.min_paragraph_chars <= para_len <= self.max_paragraph_chars
            and not paragraph.startswith('[')  # Skip citation markers
            and not paragraph.startswith('!')  # Skip info boxes
            and not paragraph.startswith('{')  # Skip templates
        )

    def _extract_sections(self, page) -> List[WikiSection]:
        """
        Extract sections with REAL NAMES from Wikipedia page.

        Uses the page.sections property which gives actual section titles.
        """
        sections = []

        # The Wikipedia library provides sections as a list of section titles
        # But we need to parse content to get paragraphs for each section
        content = page.content

        # Split by section headers (== Section Name ==)
        import re

        # Pattern to match section headers at any level
        # == Title == or === Subtitle === etc.
        section_pattern = r'^(={2,})\s*(.+?)\s*\1\s*$'

        lines = content.split('\n')
        current_section_title = None
        current_paragraphs = []

        for line in lines:
            # Check if this is a section header
            match = re.match(section_pattern, line.strip())
            if match:
                # Save previous section if it exists
                if current_section_title and current_paragraphs:
                    # Filter to only keep good paragraphs
                    valid_paras = [p for p in current_paragraphs if self._filter_paragraph(p)]
                    if valid_paras:
                        # Skip ignored sections
                        if current_section_title.lower() not in self.ignore_sections:
                            sections.append(WikiSection(
                                title=current_section_title,
                                paragraphs=valid_paras,
                                level=len(match.group(1))  # Number of = signs
                            ))

                # Start new section
                current_section_title = match.group(2).strip()
                current_paragraphs = []
            else:
                # This is content, not a header
                line = line.strip()
                if line and len(line) >= 50:  # Minimum line length
                    current_paragraphs.append(line)

        # Don't forget the last section
        if current_section_title and current_paragraphs:
            valid_paras = [p for p in current_paragraphs if self._filter_paragraph(p)]
            if valid_paras and current_section_title.lower() not in self.ignore_sections:
                sections.append(WikiSection(
                    title=current_section_title,
                    paragraphs=valid_paras,
                    level=2
                ))

        return sections

    def fetch_article(
        self,
        title: str,
        use_cache: bool = True,
        max_retries: int = 3
    ) -> Optional[WikiArticle]:
        """
        Fetch a Wikipedia article with retry logic.

        Args:
            title: Article title
            use_cache: Whether to use cached version
            max_retries: Maximum number of retry attempts

        Returns:
            WikiArticle if successful, None otherwise
        """
        # Try cache first
        if use_cache:
            cached = self._load_from_cache(title)
            if cached:
                return cached

        # Fetch from Wikipedia with retries
        for attempt in range(max_retries):
            try:
                # Search for the article to handle disambiguation
                search_results = wikipedia.search(title, results=1)
                if not search_results:
                    return None

                # Get the page
                page = wikipedia.page(search_results[0], auto_suggest=False)

                # Extract sections with REAL section names
                sections = self._extract_sections(page)

                # Validate article quality
                if len(sections) < self.min_sections_required:
                    return None

                # Create article object
                article = WikiArticle(
                    title=page.title,
                    summary=page.summary,
                    sections=sections,
                    url=page.url
                )

                # Cache for future use
                self._save_to_cache(article)

                return article

            except wikipedia.exceptions.DisambiguationError as e:
                # If disambiguation, try the first option
                if e.options and attempt == 0:
                    title = e.options[0]
                    continue
                return None

            except wikipedia.exceptions.PageError:
                # Page doesn't exist
                return None

            except Exception as e:
                # Network error or other issues - retry with backoff
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                return None

        return None

    def fetch_articles_batch(
        self,
        titles: List[str],
        use_cache: bool = True
    ) -> Dict[str, WikiArticle]:
        """
        Fetch multiple articles.

        Args:
            titles: List of article titles
            use_cache: Whether to use cached versions

        Returns:
            Dictionary mapping successful titles to WikiArticle objects
        """
        results = {}

        for title in titles:
            article = self.fetch_article(title, use_cache=use_cache)
            if article:
                results[title] = article

        return results

    def validate_article_quality(self, article: WikiArticle) -> Dict[str, Any]:
        """
        Validate article meets quality requirements.

        Returns:
            Dictionary with validation results
        """
        total_paragraphs = sum(len(section.paragraphs) for section in article.sections)

        return {
            'valid': len(article.sections) >= self.min_sections_required and total_paragraphs >= 20,
            'num_sections': len(article.sections),
            'total_paragraphs': total_paragraphs,
            'avg_paragraphs_per_section': total_paragraphs / len(article.sections) if article.sections else 0
        }
