"""
Test script to verify Wikipedia section names are extracted correctly.
Tests with 3 articles and saves to debug directory.
"""

import sys
from pathlib import Path
import json

# Add project root to path
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# Direct import to avoid core __init__ issues
import importlib.util
spec = importlib.util.spec_from_file_location(
    "fetcher_fixed",
    project_root / "core" / "wikipedia" / "fetcher_fixed.py"
)
fetcher_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetcher_module)
WikipediaFetcher = fetcher_module.WikipediaFetcher


def test_section_extraction():
    """Test that real section names are extracted from Wikipedia."""

    print("="*80)
    print("TESTING WIKIPEDIA SECTION NAME EXTRACTION")
    print("="*80)

    # Create debug directory
    debug_dir = project_root / 'data' / 'wikipedia_debug'
    debug_dir.mkdir(exist_ok=True, parents=True)

    # Initialize fetcher with debug cache
    fetcher = WikipediaFetcher(
        cache_dir=str(debug_dir / 'cache')
    )

    # Test with 3 diverse articles
    test_articles = [
        "Machine learning",
        "Climate change",
        "Photosynthesis"
    ]

    print(f"\nTesting with {len(test_articles)} articles:")
    for title in test_articles:
        print(f"  - {title}")

    print("\n" + "="*80)

    results = []

    for title in test_articles:
        print(f"\nFetching: {title}")
        print("-"*80)

        article = fetcher.fetch_article(title, use_cache=False)

        if article:
            print(f"[OK] Successfully fetched: {article.title}")
            print(f"  Sections found: {len(article.sections)}")
            print(f"\n  Section names:")

            for i, section in enumerate(article.sections[:10], 1):  # Show first 10
                para_count = len(section.paragraphs)
                print(f"    {i}. '{section.title}' ({para_count} paragraphs)")

                # Show first paragraph preview
                if section.paragraphs:
                    preview = section.paragraphs[0][:100] + "..."
                    print(f"       Preview: {preview}")

            # Check if we're getting real names or generic ones
            section_titles = [s.title for s in article.sections]
            has_generic_names = any(title.startswith("Section ") for title in section_titles)

            if has_generic_names:
                print(f"\n  [WARNING] Found generic section names (Section 1, Section 2, etc.)")
            else:
                print(f"\n  [OK] All sections have real names!")

            # Validate quality
            validation = fetcher.validate_article_quality(article)
            print(f"\n  Quality: {'VALID' if validation['valid'] else 'INVALID'}")
            print(f"    - Sections: {validation['num_sections']}")
            print(f"    - Total paragraphs: {validation['total_paragraphs']}")
            print(f"    - Avg paragraphs/section: {validation['avg_paragraphs_per_section']:.1f}")

            results.append({
                'title': article.title,
                'sections': [
                    {
                        'title': s.title,
                        'paragraph_count': len(s.paragraphs),
                        'first_paragraph_preview': s.paragraphs[0][:200] if s.paragraphs else ''
                    }
                    for s in article.sections
                ],
                'has_generic_names': has_generic_names,
                'valid': validation['valid']
            })

        else:
            print(f"[ERROR] Failed to fetch: {title}")
            results.append({
                'title': title,
                'error': 'Failed to fetch'
            })

    # Save results to JSON
    output_file = debug_dir / 'section_extraction_test.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    successful = sum(1 for r in results if 'error' not in r)
    generic_names = sum(1 for r in results if r.get('has_generic_names', False))

    print(f"Successfully fetched: {successful}/{len(test_articles)}")
    print(f"Articles with generic names: {generic_names}/{successful}")
    print(f"\nResults saved to: {output_file}")

    if generic_names == 0 and successful > 0:
        print("\n[SUCCESS] All articles have real section names!")
        print("The fix is working correctly.")
    else:
        print("\n[ISSUE] Some articles still have generic section names")
        print("The section extraction may need further debugging.")

    return results


if __name__ == "__main__":
    results = test_section_extraction()
