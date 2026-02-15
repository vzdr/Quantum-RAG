"""
Fix section names in existing Wikipedia dataset without regenerating embeddings.

This script:
1. Reads articles.jsonl to get article titles
2. Uses the fixed fetcher to get real section names
3. Creates mapping from "Section N" to real section names
4. Updates articles.jsonl and chunks.jsonl with correct names
5. Preserves all embeddings and other data
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

# Add project root to path
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# Import the fixed fetcher
import importlib.util
spec = importlib.util.spec_from_file_location(
    "fetcher",
    project_root / "core" / "wikipedia" / "fetcher.py"
)
fetcher_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetcher_module)
WikipediaFetcher = fetcher_module.WikipediaFetcher


def load_jsonl(filepath: Path) -> List[Dict]:
    """Load JSONL file into list of dictionaries."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], filepath: Path, backup: bool = True):
    """Save list of dictionaries to JSONL file."""
    if backup and filepath.exists():
        backup_path = filepath.with_suffix('.jsonl.backup')
        print(f"Creating backup: {backup_path}")
        filepath.rename(backup_path)

    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def get_section_name_mapping(article_title: str, fetcher: WikipediaFetcher) -> Dict[str, str]:
    """
    Get mapping from generic 'Section N' to real section names.

    Args:
        article_title: Title of Wikipedia article
        fetcher: WikipediaFetcher instance with fixed section extraction

    Returns:
        Dictionary mapping "Section 1" -> real section name
    """
    # Fetch article with fixed fetcher
    article = fetcher.fetch_article(article_title, use_cache=True)

    if not article:
        print(f"  [WARNING] Could not fetch article: {article_title}")
        return {}

    # Create mapping based on order
    mapping = {}
    for i, section in enumerate(article.sections, 1):
        generic_name = f"Section {i}"
        real_name = section.title
        mapping[generic_name] = real_name

    return mapping


def fix_article_section_names(
    articles_data: List[Dict],
    fetcher: WikipediaFetcher
) -> Tuple[List[Dict], Dict[str, Dict[str, str]]]:
    """
    Fix section names in articles data.

    Returns:
        Tuple of (updated articles data, mapping dict for each article)
    """
    updated_articles = []
    all_mappings = {}

    print("\n" + "="*80)
    print("FIXING ARTICLE SECTION NAMES")
    print("="*80)

    for article in tqdm(articles_data, desc="Processing articles", unit="article"):
        article_title = article['title']

        # Get section name mapping
        mapping = get_section_name_mapping(article_title, fetcher)

        if not mapping:
            print(f"\n[WARNING] No mapping found for: {article_title}")
            updated_articles.append(article)
            continue

        all_mappings[article_title] = mapping

        # Update section names in article
        updated_article = article.copy()
        updated_sections = []

        for section in article['sections']:
            updated_section = section.copy()
            old_name = section['title']
            new_name = mapping.get(old_name, old_name)
            updated_section['title'] = new_name
            updated_sections.append(updated_section)

        updated_article['sections'] = updated_sections
        updated_articles.append(updated_article)

    return updated_articles, all_mappings


def fix_chunk_section_names(
    chunks_data: List[Dict],
    article_mappings: Dict[str, Dict[str, str]]
) -> List[Dict]:
    """
    Fix section names in chunks data using article mappings.

    Args:
        chunks_data: List of chunk dictionaries
        article_mappings: Dict mapping article_title -> (generic_name -> real_name)

    Returns:
        Updated chunks data
    """
    updated_chunks = []

    print("\n" + "="*80)
    print("FIXING CHUNK SECTION NAMES")
    print("="*80)

    updates_made = 0

    for chunk in tqdm(chunks_data, desc="Processing chunks", unit="chunk"):
        updated_chunk = chunk.copy()

        # Only update chunks that have section_title
        if 'section_title' in chunk and 'article_title' in chunk:
            article_title = chunk['article_title']
            old_section_name = chunk['section_title']

            # Get mapping for this article
            if article_title in article_mappings:
                mapping = article_mappings[article_title]
                new_section_name = mapping.get(old_section_name, old_section_name)

                if new_section_name != old_section_name:
                    updated_chunk['section_title'] = new_section_name
                    updates_made += 1

        updated_chunks.append(updated_chunk)

    print(f"\nUpdated {updates_made} chunk section names")
    return updated_chunks


def verify_updates(
    articles_data: List[Dict],
    chunks_data: List[Dict]
):
    """Verify that section names were updated correctly."""
    print("\n" + "="*80)
    print("VERIFICATION")
    print("="*80)

    # Check articles
    articles_with_generic = 0
    total_sections = 0

    for article in articles_data:
        for section in article['sections']:
            total_sections += 1
            if section['title'].startswith('Section '):
                articles_with_generic += 1

    print(f"\nArticles:")
    print(f"  Total sections: {total_sections}")
    print(f"  Generic names remaining: {articles_with_generic}")
    print(f"  Real names: {total_sections - articles_with_generic}")

    # Check chunks
    chunks_with_generic = 0
    chunks_with_section = 0

    for chunk in chunks_data:
        if 'section_title' in chunk:
            chunks_with_section += 1
            if chunk['section_title'].startswith('Section '):
                chunks_with_generic += 1

    print(f"\nChunks:")
    print(f"  Chunks with section_title: {chunks_with_section}")
    print(f"  Generic names remaining: {chunks_with_generic}")
    print(f"  Real names: {chunks_with_section - chunks_with_generic}")

    if articles_with_generic == 0 and chunks_with_generic == 0:
        print("\n[SUCCESS] All section names updated to real names!")
    else:
        print("\n[WARNING] Some generic section names remain")


def main():
    """Main execution function."""
    print("="*80)
    print("WIKIPEDIA DATASET SECTION NAME FIXER")
    print("="*80)
    print("\nThis script updates section names in articles.jsonl and chunks.jsonl")
    print("without regenerating embeddings or other data.")

    # Paths
    data_dir = project_root / 'data' / 'wikipedia'
    checkpoints_dir = data_dir / 'checkpoints'
    articles_path = checkpoints_dir / 'articles.jsonl'
    chunks_path = checkpoints_dir / 'chunks.jsonl'

    if not articles_path.exists():
        print(f"\n[ERROR] Articles file not found: {articles_path}")
        return

    if not chunks_path.exists():
        print(f"\n[ERROR] Chunks file not found: {chunks_path}")
        return

    # Initialize fetcher with fixed section extraction
    cache_dir = data_dir / 'cache'
    fetcher = WikipediaFetcher(cache_dir=str(cache_dir))

    print(f"\nUsing cache directory: {cache_dir}")

    # Load existing data
    print("\nLoading existing data...")
    articles_data = load_jsonl(articles_path)
    chunks_data = load_jsonl(chunks_path)

    print(f"  Loaded {len(articles_data)} articles")
    print(f"  Loaded {len(chunks_data)} chunks")

    # Fix article section names
    updated_articles, article_mappings = fix_article_section_names(articles_data, fetcher)

    # Show sample mapping
    if article_mappings:
        sample_article = list(article_mappings.keys())[0]
        sample_mapping = article_mappings[sample_article]
        print(f"\nSample mapping for '{sample_article}':")
        for old_name, new_name in list(sample_mapping.items())[:5]:
            print(f"  '{old_name}' -> '{new_name}'")

    # Fix chunk section names
    updated_chunks = fix_chunk_section_names(chunks_data, article_mappings)

    # Verify updates
    verify_updates(updated_articles, updated_chunks)

    # Save updated data (with backups)
    print("\n" + "="*80)
    print("SAVING UPDATED DATA")
    print("="*80)

    save_jsonl(updated_articles, articles_path, backup=True)
    print(f"[OK] Saved updated articles to: {articles_path}")

    save_jsonl(updated_chunks, chunks_path, backup=True)
    print(f"[OK] Saved updated chunks to: {chunks_path}")

    print("\n[SUCCESS] Section names updated successfully!")
    print("Original files backed up with .backup extension")


if __name__ == "__main__":
    main()
