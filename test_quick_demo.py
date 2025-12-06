"""
Quick Demo Script - Wikipedia Dataset

A simple script to demonstrate the Wikipedia dataset integration.
Shows basic retrieval and comparison between different methods.

This is a quick way to verify everything works and see the system in action.
"""

from core.vector_store import VectorStore
from core.embedder import EmbeddingGenerator
from core.analysis_utils import evaluate_retrieval_quality
import numpy as np

def print_header(text):
    """Print a nice header."""
    print("\n" + "="*80)
    print(text.center(80))
    print("="*80 + "\n")

def main():
    print_header("WIKIPEDIA DATASET QUICK DEMO")

    # Step 1: Initialize
    print("📊 Initializing Wikipedia dataset...")
    vector_store = VectorStore(
        collection_name="wiki_aspects",
        persist_directory="./data/wikipedia/chroma_db"
    )

    print(f"✓ Loaded ChromaDB with {vector_store.count:,} chunks")

    print("\n🤖 Loading embedding model (BGE-large-en-v1.5)...")
    embedder = EmbeddingGenerator(
        model_name="BAAI/bge-large-en-v1.5",
        device="cpu"
    )
    print(f"✓ Model loaded (dimension: {embedder.embedding_dim})")

    # Step 2: Show dataset statistics
    print_header("DATASET STATISTICS")

    # Count different chunk types
    gold_base = vector_store.get_by_metadata({"chunk_type": "gold_base"})
    gold_redundant = vector_store.get_by_metadata({"chunk_type": "gold_redundant"})
    noise = vector_store.get_by_metadata({"chunk_type": "noise"})
    prompts = vector_store.get_by_metadata({"chunk_type": "prompt"})

    print(f"Total chunks:      {vector_store.count:,}")
    print(f"  Gold base:       {len(gold_base):,}")
    print(f"  Gold redundant:  {len(gold_redundant):,}")
    print(f"  Noise:           {len(noise):,}")
    print(f"  Prompts:         {len(prompts):,}")

    # Step 3: Test retrieval
    print_header("RETRIEVAL DEMONSTRATION")

    query = "What were the major achievements of ancient Roman civilization?"
    print(f"Query: '{query}'\n")

    # Embed query
    query_embedding = embedder.embed_query(query)

    # Test 1: Basic retrieval (all chunks)
    print("1️⃣  Basic Retrieval (Top-5, all chunks):")
    print("-" * 80)
    results_all = vector_store.search(query_embedding, k=5)

    for i, result in enumerate(results_all, 1):
        chunk_type = result['metadata'].get('chunk_type', 'unknown')
        aspect = result['metadata'].get('aspect_name', 'N/A')
        redundancy = result['metadata'].get('redundancy_index', 'N/A')

        print(f"\n  [{i}] Score: {result['score']:.4f} | Type: {chunk_type} | Aspect: {aspect} | Redundancy: {redundancy}")
        print(f"      {result['text'][:150]}...")

    # Test 2: Filter out noise
    print("\n\n2️⃣  Filtered Retrieval (Top-5, excluding noise):")
    print("-" * 80)
    results_filtered = vector_store.search_with_filters(
        query_embedding,
        k=5,
        exclude_metadata={"chunk_type": "noise"}
    )

    for i, result in enumerate(results_filtered, 1):
        chunk_type = result['metadata'].get('chunk_type', 'unknown')
        aspect = result['metadata'].get('aspect_name', 'N/A')
        redundancy = result['metadata'].get('redundancy_index', 'N/A')

        print(f"\n  [{i}] Score: {result['score']:.4f} | Type: {chunk_type} | Aspect: {aspect} | Redundancy: {redundancy}")
        print(f"      {result['text'][:150]}...")

    # Test 3: Only base chunks (no redundancy)
    print("\n\n3️⃣  Base Chunks Only (Top-5, gold base chunks):")
    print("-" * 80)
    results_base = vector_store.search_with_filters(
        query_embedding,
        k=5,
        metadata_filter={"chunk_type": "gold_base"}
    )

    for i, result in enumerate(results_base, 1):
        aspect = result['metadata'].get('aspect_name', 'N/A')
        article = result['metadata'].get('article_title', 'N/A')

        print(f"\n  [{i}] Score: {result['score']:.4f} | Aspect: {aspect}")
        print(f"      Article: {article}")
        print(f"      {result['text'][:150]}...")

    # Step 4: Diversity analysis
    print_header("DIVERSITY ANALYSIS")

    # Compare diversity of different retrieval strategies
    print("Comparing intra-list similarity (lower = more diverse):\n")

    def compute_diversity(results):
        """Compute average pairwise similarity."""
        if len(results) < 2:
            return 0.0

        embeddings = np.array([r['embedding'] for r in results if r.get('embedding')])
        if len(embeddings) < 2:
            return 0.0

        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-8)

        # Pairwise similarities
        similarities = []
        for i in range(len(normalized)):
            for j in range(i+1, len(normalized)):
                sim = np.dot(normalized[i], normalized[j])
                similarities.append(sim)

        return np.mean(similarities) if similarities else 0.0

    # Get results with embeddings
    results_all_emb = vector_store.search(query_embedding, k=10)
    results_filtered_emb = vector_store.search_with_filters(
        query_embedding, k=10,
        exclude_metadata={"chunk_type": "noise"}
    )
    results_base_emb = vector_store.search_with_filters(
        query_embedding, k=10,
        metadata_filter={"chunk_type": "gold_base"}
    )

    div_all = compute_diversity(results_all_emb)
    div_filtered = compute_diversity(results_filtered_emb)
    div_base = compute_diversity(results_base_emb)

    print(f"  All chunks:        {div_all:.4f}")
    print(f"  Filtered (no noise): {div_filtered:.4f}")
    print(f"  Base only:         {div_base:.4f}")

    print("\n💡 Lower values indicate more diverse results")

    # Step 5: Show chunk types distribution
    print_header("RETRIEVAL COMPOSITION")

    def count_types(results):
        counts = {}
        for r in results:
            chunk_type = r['metadata'].get('chunk_type', 'unknown')
            counts[chunk_type] = counts.get(chunk_type, 0) + 1
        return counts

    print("Distribution of chunk types in Top-10 results:\n")

    types_all = count_types(results_all_emb)
    types_filtered = count_types(results_filtered_emb)
    types_base = count_types(results_base_emb)

    print("All chunks:")
    for ctype, count in sorted(types_all.items()):
        print(f"  {ctype:20s}: {count:2d}")

    print("\nFiltered (no noise):")
    for ctype, count in sorted(types_filtered.items()):
        print(f"  {ctype:20s}: {count:2d}")

    print("\nBase only:")
    for ctype, count in sorted(types_base.items()):
        print(f"  {ctype:20s}: {count:2d}")

    # Final message
    print_header("DEMO COMPLETE")

    print("✓ All tests completed successfully!")
    print("\nWhat you just saw:")
    print("  1. Loaded 5,600 Wikipedia chunks from ChromaDB")
    print("  2. Tested retrieval with different filters")
    print("  3. Analyzed diversity of results")
    print("  4. Examined chunk type distributions")

    print("\n📖 Next steps:")
    print("  - Read WIKIPEDIA_INTEGRATION.md for detailed usage")
    print("  - Open RAG_System.ipynb for interactive exploration")
    print("  - Try different queries and filters")
    print("  - Experiment with QUBO-based retrieval strategies")

    print("\n" + "="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("  1. Run 'python test_integration.py' to diagnose issues")
        print("  2. Ensure virtual environment is activated")
        print("  3. Check that dataset files were copied correctly")
