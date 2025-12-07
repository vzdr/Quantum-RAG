"""
Clean invalid Unicode characters from chunks.jsonl and articles.jsonl while preserving all valid text.

This script:
1. Reads chunks.jsonl and articles.jsonl line by line
2. Cleans invalid Unicode characters from text fields
3. Writes cleaned data back to files with backups
"""

import json
import sys
import unicodedata
from pathlib import Path
from tqdm import tqdm


def clean_unicode_string(text: str) -> str:
    """
    Clean invalid Unicode characters from a string.

    This removes:
    - Control characters (except newlines, tabs)
    - Invalid Unicode sequences
    - Problematic characters that can't be encoded in common formats

    But preserves all meaningful text content.
    """
    if not isinstance(text, str):
        return text

    # Replace common problematic characters
    # Minus sign (U+2212) -> hyphen-minus (U+002D)
    text = text.replace('\u2212', '-')

    # En dash (U+2013) and Em dash (U+2014) -> regular dash
    text = text.replace('\u2013', '-')
    text = text.replace('\u2014', '--')

    # Various quotes to regular quotes
    text = text.replace('\u2018', "'")  # Left single quote
    text = text.replace('\u2019', "'")  # Right single quote
    text = text.replace('\u201c', '"')  # Left double quote
    text = text.replace('\u201d', '"')  # Right double quote

    # Ellipsis to three dots
    text = text.replace('\u2026', '...')

    # Non-breaking space to regular space
    text = text.replace('\u00a0', ' ')

    # Remove other control characters except newlines and tabs
    cleaned = []
    for char in text:
        # Keep printable characters, newlines, and tabs
        if char in ('\n', '\r', '\t'):
            cleaned.append(char)
        elif unicodedata.category(char)[0] == 'C':
            # Skip other control characters
            continue
        else:
            # Keep all other characters
            cleaned.append(char)

    return ''.join(cleaned)


def clean_chunk(chunk: dict) -> dict:
    """
    Clean all string fields in a chunk dictionary.

    Args:
        chunk: Chunk dictionary with potential Unicode issues

    Returns:
        Cleaned chunk dictionary
    """
    cleaned = {}

    for key, value in chunk.items():
        if isinstance(value, str):
            cleaned[key] = clean_unicode_string(value)
        elif isinstance(value, dict):
            cleaned[key] = clean_chunk(value)
        elif isinstance(value, list):
            cleaned[key] = [
                clean_unicode_string(item) if isinstance(item, str) else item
                for item in value
            ]
        else:
            cleaned[key] = value

    return cleaned


def clean_jsonl_file(filepath: Path, backup_suffix: str = '.backup3'):
    """
    Clean Unicode characters in a JSONL file.

    Args:
        filepath: Path to JSONL file
        backup_suffix: Suffix for backup file
    """
    print(f"\n{'='*80}")
    print(f"Processing: {filepath.name}")
    print('='*80)

    if not filepath.exists():
        print(f"[ERROR] File not found: {filepath}")
        return

    # Create backup
    backup_path = filepath.parent / (filepath.stem + backup_suffix)
    print(f"Creating backup: {backup_path}")

    with open(filepath, 'rb') as f:
        original_content = f.read()

    with open(backup_path, 'wb') as f:
        f.write(original_content)

    print(f"[OK] Backup created")

    # Load data
    print("\nLoading data...")
    items = []
    errors = 0

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                items.append(item)
            except json.JSONDecodeError as e:
                print(f"[WARNING] JSON decode error on line {line_num}: {e}")
                errors += 1

    print(f"Loaded {len(items)} items ({errors} errors)")

    # Clean items
    print("Cleaning Unicode characters...")
    cleaned_items = []

    for item in tqdm(items, desc=f"Cleaning {filepath.name}", unit="item"):
        cleaned_item = clean_chunk(item)
        cleaned_items.append(cleaned_item)

    # Save cleaned items
    print("Saving cleaned data...")
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in cleaned_items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"[OK] Saved {len(cleaned_items)} cleaned items")

    return items, cleaned_items


def main():
    """Main execution function."""
    print("="*80)
    print("UNICODE CLEANER FOR WIKIPEDIA DATA")
    print("="*80)
    print("\nThis script cleans invalid Unicode characters from chunks.jsonl and articles.jsonl")
    print("while preserving all meaningful text content.")

    # Get script directory and project root
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent

    # Paths
    data_dir = project_root / 'data' / 'wikipedia'
    checkpoints_dir = data_dir / 'checkpoints'
    chunks_path = checkpoints_dir / 'chunks.jsonl'
    articles_path = checkpoints_dir / 'articles.jsonl'

    all_original_chars = set()
    all_cleaned_chars = set()

    # Clean chunks.jsonl
    chunks_original, chunks_cleaned = clean_jsonl_file(chunks_path, '.backup3')

    # Collect character stats
    for chunk in chunks_original:
        for key, value in chunk.items():
            if isinstance(value, str):
                all_original_chars.update(value)

    for chunk in chunks_cleaned:
        for key, value in chunk.items():
            if isinstance(value, str):
                all_cleaned_chars.update(value)

    # Clean articles.jsonl
    articles_original, articles_cleaned = clean_jsonl_file(articles_path, '.backup3')

    # Collect character stats
    def extract_chars(obj):
        """Recursively extract all characters from nested dict/list structure."""
        chars = set()
        if isinstance(obj, str):
            chars.update(obj)
        elif isinstance(obj, dict):
            for value in obj.values():
                chars.update(extract_chars(value))
        elif isinstance(obj, list):
            for item in obj:
                chars.update(extract_chars(item))
        return chars

    for article in articles_original:
        all_original_chars.update(extract_chars(article))

    for article in articles_cleaned:
        all_cleaned_chars.update(extract_chars(article))

    # Statistics
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)

    removed_chars = all_original_chars - all_cleaned_chars

    print(f"\nUnique characters in original: {len(all_original_chars)}")
    print(f"Unique characters after cleaning: {len(all_cleaned_chars)}")
    print(f"Characters removed/replaced: {len(removed_chars)}")

    if removed_chars:
        print(f"\nRemoved characters (showing first 30):")
        for i, char in enumerate(sorted(removed_chars)[:30]):
            try:
                char_name = unicodedata.name(char, 'UNKNOWN')
                print(f"  U+{ord(char):04X} ({char!r}): {char_name}")
            except:
                print(f"  U+{ord(char):04X} ({char!r})")

    print("\n[SUCCESS] Unicode cleaning complete for both files!")


if __name__ == "__main__":
    main()
