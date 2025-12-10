"""
Fix aspect names in chunks.jsonl to use real section names.

This script:
1. Loads the updated articles.jsonl with real section names
2. Creates mapping from "Section N" to real section names per article
3. Updates aspect_name field in chunks.jsonl
4. Updates prompt text to reference real section names
5. Preserves all embeddings and other data
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

# Add project root to path
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent
sys.path.insert(0, str(project_root))


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
        backup_path = filepath.with_suffix('.jsonl.backup2')
        print(f"Creating backup: {backup_path}")
        filepath.rename(backup_path)

    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def build_section_mappings(articles_data: List[Dict]) -> Dict[str, Dict[str, str]]:
    """
    Build mapping from article_title -> (generic_name -> real_name).

    Args:
        articles_data: List of article dictionaries with sections

    Returns:
        Dict mapping article_title to section name mapping
    """
    mappings = {}

    for article in articles_data:
        article_title = article['title']
        mapping = {}

        for i, section in enumerate(article['sections'], 1):
            generic_name = f"Section {i}"
            real_name = section['title']
            mapping[generic_name] = real_name

        mappings[article_title] = mapping

    return mappings


def fix_chunk_aspect_names(
    chunks_data: List[Dict],
    section_mappings: Dict[str, Dict[str, str]]
) -> List[Dict]:
    """
    Fix aspect_name field in chunks and update prompt text.

    Args:
        chunks_data: List of chunk dictionaries
        section_mappings: Dict mapping article_title -> (generic_name -> real_name)

    Returns:
        Updated chunks data
    """
    updated_chunks = []
    aspect_name_updates = 0
    prompt_updates = 0

    print("\n" + "="*80)
    print("FIXING CHUNK ASPECT NAMES AND PROMPTS")
    print("="*80)

    for chunk in tqdm(chunks_data, desc="Processing chunks", unit="chunk"):
        updated_chunk = chunk.copy()

        article_title = chunk.get('article_title')
        if not article_title or article_title not in section_mappings:
            updated_chunks.append(updated_chunk)
            continue

        mapping = section_mappings[article_title]

        # Fix aspect_name if it's a generic section name
        aspect_name = chunk.get('aspect_name', '')
        if aspect_name.startswith('Section ') and aspect_name in mapping:
            updated_chunk['aspect_name'] = mapping[aspect_name]
            aspect_name_updates += 1

        # Fix prompt text if it contains generic section names
        if chunk.get('chunk_type') == 'prompt':
            text = chunk.get('text', '')

            # Check if text contains generic section names
            if 'Section 1' in text or 'Section 2' in text:
                # Get all section names for this article
                section_names = [mapping.get(f"Section {i}", f"Section {i}")
                               for i in range(1, len(mapping) + 1)]

                # Build new prompt text
                # Original format: "Provide a comprehensive overview of {article}, covering key aspects such as Section 1, Section 2, Section 3."
                new_text = f"Provide a comprehensive overview of {article_title}, covering key aspects such as {', '.join(section_names[:5])}."

                updated_chunk['text'] = new_text
                prompt_updates += 1

        updated_chunks.append(updated_chunk)

    print(f"\nUpdated {aspect_name_updates} aspect names")
    print(f"Updated {prompt_updates} prompts")

    return updated_chunks


def verify_updates(chunks_data: List[Dict]):
    """Verify that aspect names and prompts were updated correctly."""
    print("\n" + "="*80)
    print("VERIFICATION")
    print("="*80)

    # Check aspect names
    chunks_with_generic_aspect = 0
    chunks_with_section_aspect = 0

    for chunk in chunks_data:
        aspect_name = chunk.get('aspect_name', '')
        if aspect_name.startswith('Section ') and aspect_name != 'prompt':
            chunks_with_generic_aspect += 1
        elif aspect_name not in ['prompt', '']:
            chunks_with_section_aspect += 1

    print(f"\nAspect Names:")
    print(f"  Chunks with real aspect names: {chunks_with_section_aspect}")
    print(f"  Chunks with generic aspect names: {chunks_with_generic_aspect}")

    # Check prompts
    prompts_with_generic = 0
    total_prompts = 0

    for chunk in chunks_data:
        if chunk.get('chunk_type') == 'prompt':
            total_prompts += 1
            text = chunk.get('text', '')
            if 'Section 1' in text or 'Section 2' in text:
                prompts_with_generic += 1

    print(f"\nPrompts:")
    print(f"  Total prompts: {total_prompts}")
    print(f"  Prompts with generic section names: {prompts_with_generic}")
    print(f"  Prompts with real section names: {total_prompts - prompts_with_generic}")

    if chunks_with_generic_aspect == 0 and prompts_with_generic == 0:
        print("\n[SUCCESS] All aspect names and prompts updated to real section names!")
    else:
        print("\n[WARNING] Some generic section names remain")


def main():
    """Main execution function."""
    print("="*80)
    print("CHUNK ASPECT NAME FIXER")
    print("="*80)
    print("\nThis script updates aspect_name fields and prompt text in chunks.jsonl")
    print("to use real section names instead of generic 'Section N' names.")

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

    # Load data
    print("\nLoading data...")
    articles_data = load_jsonl(articles_path)
    chunks_data = load_jsonl(chunks_path)

    print(f"  Loaded {len(articles_data)} articles")
    print(f"  Loaded {len(chunks_data)} chunks")

    # Build section mappings from articles
    print("\nBuilding section name mappings from articles...")
    section_mappings = build_section_mappings(articles_data)
    print(f"  Created mappings for {len(section_mappings)} articles")

    # Show sample mapping
    if section_mappings:
        sample_article = list(section_mappings.keys())[0]
        sample_mapping = section_mappings[sample_article]
        print(f"\nSample mapping for '{sample_article}':")
        for old_name, new_name in list(sample_mapping.items())[:5]:
            print(f"  '{old_name}' -> '{new_name}'")

    # Fix chunk aspect names and prompts
    updated_chunks = fix_chunk_aspect_names(chunks_data, section_mappings)

    # Verify updates
    verify_updates(updated_chunks)

    # Save updated data (with backup)
    print("\n" + "="*80)
    print("SAVING UPDATED DATA")
    print("="*80)

    save_jsonl(updated_chunks, chunks_path, backup=True)
    print(f"[OK] Saved updated chunks to: {chunks_path}")

    print("\n[SUCCESS] Chunk aspect names and prompts updated successfully!")
    print("Original file backed up with .backup2 extension")


if __name__ == "__main__":
    main()
