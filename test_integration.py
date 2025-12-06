"""
Integration Test Script for Wikipedia Dataset

This script validates that the Wikipedia dataset integration is working correctly.
It tests all key components and provides a clear pass/fail report.

Run this script after integration to ensure everything is set up properly.
"""

import sys
from pathlib import Path
import numpy as np

def test_imports():
    """Test that all required modules can be imported."""
    print("\n" + "="*80)
    print("TEST 1: Importing Modules")
    print("="*80)

    try:
        from core.vector_store import VectorStore
        print("✓ VectorStore imported")

        from core.embedder import EmbeddingGenerator
        print("✓ EmbeddingGenerator imported")

        from core.wikipedia.fetcher import WikipediaFetcher, WikiArticle
        print("✓ Wikipedia fetcher imported")

        from core.analysis_utils import (
            compute_pairwise_similarities,
            evaluate_retrieval_quality,
            compute_qubo_energy
        )
        print("✓ Analysis utilities imported")

        from config.settings import RAGConfig
        print("✓ RAGConfig imported")

        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_dataset_files():
    """Test that Wikipedia dataset files exist."""
    print("\n" + "="*80)
    print("TEST 2: Checking Dataset Files")
    print("="*80)

    base_path = Path("./data/wikipedia")

    checks = {
        "ChromaDB directory": base_path / "chroma_db",
        "Cache directory": base_path / "cache",
        "Checkpoints directory": base_path / "checkpoints",
        "Similarity directory": base_path / "similarity",
        "Wiki articles list": base_path / "wiki_articles.txt",
        "Similarity matrix": base_path / "similarity" / "similarity_matrix.npz",
    }

    all_exist = True
    for name, path in checks.items():
        if path.exists():
            if path.is_file():
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"✓ {name}: {size_mb:.1f} MB")
            else:
                print(f"✓ {name}: exists")
        else:
            print(f"✗ {name}: NOT FOUND at {path}")
            all_exist = False

    return all_exist


def test_chromadb_loading():
    """Test loading the Wikipedia ChromaDB."""
    print("\n" + "="*80)
    print("TEST 3: Loading ChromaDB")
    print("="*80)

    try:
        from core.vector_store import VectorStore

        vector_store = VectorStore(
            collection_name="wiki_aspects",
            persist_directory="./data/wikipedia/chroma_db"
        )

        count = vector_store.count
        print(f"✓ ChromaDB loaded successfully")
        print(f"  Total chunks: {count}")

        if count == 0:
            print("✗ WARNING: ChromaDB is empty!")
            return False

        # Get statistics
        stats = vector_store.get_statistics()
        print(f"  Collection: {stats['collection_name']}")
        print(f"  Unique sources: {stats['unique_sources']}")

        # Test metadata query
        sample = vector_store.get_by_metadata({"chunk_type": "gold_base"})
        print(f"  Gold base chunks: {len(sample)}")

        return True

    except Exception as e:
        print(f"✗ ChromaDB loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_embedder():
    """Test the embedding generator with BGE-large model."""
    print("\n" + "="*80)
    print("TEST 4: Testing Embedder")
    print("="*80)

    try:
        from core.embedder import EmbeddingGenerator

        print("Loading BGE-large model (this may take a moment)...")
        embedder = EmbeddingGenerator(
            model_name="BAAI/bge-large-en-v1.5",
            device="cpu"
        )

        # Test embedding a query
        query = "What is the history of Rome?"
        query_embedding = embedder.embed_query(query)

        print(f"✓ Embedder loaded successfully")
        print(f"  Model: BAAI/bge-large-en-v1.5")
        print(f"  Embedding dimension: {query_embedding.shape[0]}")
        print(f"  Sample query embedded: '{query}'")

        if query_embedding.shape[0] != 1024:
            print(f"✗ WARNING: Expected 1024 dimensions, got {query_embedding.shape[0]}")
            return False

        return True

    except Exception as e:
        print(f"✗ Embedder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_retrieval():
    """Test end-to-end retrieval."""
    print("\n" + "="*80)
    print("TEST 5: Testing Retrieval Pipeline")
    print("="*80)

    try:
        from core.vector_store import VectorStore
        from core.embedder import EmbeddingGenerator

        # Initialize
        vector_store = VectorStore(
            collection_name="wiki_aspects",
            persist_directory="./data/wikipedia/chroma_db"
        )

        embedder = EmbeddingGenerator(
            model_name="BAAI/bge-large-en-v1.5",
            device="cpu"
        )

        # Test query
        query = "Tell me about ancient civilizations"
        query_embedding = embedder.embed_query(query)

        # Basic search
        results = vector_store.search(query_embedding, k=5)

        print(f"✓ Basic search successful")
        print(f"  Query: '{query}'")
        print(f"  Results returned: {len(results)}")

        if results:
            print(f"\n  Top result:")
            print(f"    Score: {results[0]['score']:.4f}")
            print(f"    Text preview: {results[0]['text'][:100]}...")
            print(f"    Metadata: chunk_type={results[0]['metadata'].get('chunk_type', 'N/A')}")

        # Test filtered search
        filtered_results = vector_store.search_with_filters(
            query_embedding,
            k=5,
            metadata_filter={"chunk_type": "gold_base"}
        )

        print(f"\n✓ Filtered search successful")
        print(f"  Gold base results: {len(filtered_results)}")

        return True

    except Exception as e:
        print(f"✗ Retrieval test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_analysis_utils():
    """Test analysis utilities."""
    print("\n" + "="*80)
    print("TEST 6: Testing Analysis Utilities")
    print("="*80)

    try:
        from core.analysis_utils import (
            compute_pairwise_similarities,
            evaluate_retrieval_quality,
            compute_qubo_energy
        )

        # Create sample data
        embeddings = np.random.randn(10, 1024)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Test similarity computation
        sim_matrix = compute_pairwise_similarities(embeddings)
        print(f"✓ Similarity matrix computed: shape {sim_matrix.shape}")

        # Test quality evaluation
        selected = [0, 1, 2, 3, 4]
        gold = [0, 2, 4, 6, 8]

        metrics = evaluate_retrieval_quality(selected, gold, sim_matrix)
        print(f"✓ Quality metrics computed:")
        print(f"    Gold recall: {metrics['gold_recall']:.1f}%")
        print(f"    Avg redundancy: {metrics['avg_redundancy']:.3f}")

        # Test QUBO energy
        query_sims = np.random.rand(10)
        energy = compute_qubo_energy(
            query_sims, selected, sim_matrix,
            alpha=0.7, beta=1.0, K=5
        )
        print(f"✓ QUBO energy computed: {energy:.3f}")

        return True

    except Exception as e:
        print(f"✗ Analysis utilities test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_similarity_matrix():
    """Test loading precomputed similarity matrix."""
    print("\n" + "="*80)
    print("TEST 7: Testing Similarity Matrix")
    print("="*80)

    try:
        sim_path = Path("./data/wikipedia/similarity/similarity_matrix.npz")

        if not sim_path.exists():
            print(f"✗ Similarity matrix not found at {sim_path}")
            return False

        data = np.load(sim_path)
        sim_matrix = data['similarity_matrix']

        print(f"✓ Similarity matrix loaded")
        print(f"  Shape: {sim_matrix.shape}")
        print(f"  Min similarity: {sim_matrix.min():.3f}")
        print(f"  Max similarity: {sim_matrix.max():.3f}")
        print(f"  Mean similarity: {sim_matrix.mean():.3f}")

        if sim_matrix.shape[0] != sim_matrix.shape[1]:
            print(f"✗ WARNING: Matrix is not square!")
            return False

        return True

    except Exception as e:
        print(f"✗ Similarity matrix test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "#"*80)
    print("#" + " "*78 + "#")
    print("#  WIKIPEDIA DATASET INTEGRATION TEST SUITE".center(78) + " #")
    print("#" + " "*78 + "#")
    print("#"*80)

    tests = [
        ("Import Modules", test_imports),
        ("Dataset Files", test_dataset_files),
        ("ChromaDB Loading", test_chromadb_loading),
        ("Embedder (BGE-large)", test_embedder),
        ("Retrieval Pipeline", test_retrieval),
        ("Analysis Utilities", test_analysis_utils),
        ("Similarity Matrix", test_similarity_matrix),
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n✗ {name} crashed: {e}")
            results[name] = False

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(results.values())
    total = len(results)

    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print("\n" + "="*80)
    print(f"Results: {passed}/{total} tests passed")
    print("="*80)

    if passed == total:
        print("\n🎉 SUCCESS! All integration tests passed.")
        print("The Wikipedia dataset is ready to use.")
        print("\nNext steps:")
        print("  1. Run 'python test_quick_demo.py' for a quick demo")
        print("  2. Open RAG_System.ipynb to explore the system")
        print("  3. Check WIKIPEDIA_INTEGRATION.md for usage examples")
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED")
        print("Please check the errors above and ensure:")
        print("  1. The virtual environment is activated")
        print("  2. All files were copied correctly from Quantum Dice v2")
        print("  3. Dependencies are installed (pip install -r requirements.txt)")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
