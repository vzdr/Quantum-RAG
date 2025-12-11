"""
Comprehensive test to identify the exact failure point in compare endpoint.
This simulates what the API does step by step.
"""
import sys
from pathlib import Path

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

print("=" * 60)
print("Testing Compare Endpoint Flow")
print("=" * 60)

try:
    print("\n[1/6] Importing services...")
    from demo_app.backend.services.retrieval_service import get_retrieval_service
    from demo_app.backend.services.visualization_service import get_visualization_service
    print("[OK] Imports successful")

    print("\n[2/6] Initializing services...")
    retrieval_service = get_retrieval_service()
    viz_service = get_visualization_service()
    print("[OK] Services initialized")

    # Test with medical dataset and a simple query
    dataset = "medical"
    query = "What are the symptoms of lupus?"
    k = 5

    print(f"\n[3/6] Checking dataset availability...")
    available = retrieval_service.get_available_datasets()
    print(f"  Available datasets: {available}")
    if dataset not in available:
        print(f"  [ERROR] Dataset '{dataset}' not available!")
        sys.exit(1)
    print(f"[OK] Dataset '{dataset}' is available")

    print(f"\n[4/6] Running retrieval comparison...")
    print(f"  Query: {query}")
    print(f"  Dataset: {dataset}")
    print(f"  k: {k}")

    import asyncio
    results = asyncio.run(retrieval_service.compare_methods(
        query=query,
        dataset=dataset,
        k=k,
        include_llm=False,  # Skip LLM for faster testing
        alpha=0.15,
        penalty=1000.0,
        lambda_param=0.5,
        solver_preset="balanced",
    ))
    print("[OK] Retrieval comparison completed")
    print(f"  - Top-K: {len(results['topk'].results)} results")
    print(f"  - MMR: {len(results['mmr'].results)} results")
    print(f"  - QUBO: {len(results['qubo'].results)} results")

    print(f"\n[5/6] Getting UMAP embeddings...")
    try:
        umap_points = viz_service.get_embeddings_for_dataset(dataset)
        print(f"[OK] UMAP computation completed ({len(umap_points)} points)")
    except Exception as e:
        print(f"[ERROR] UMAP computation failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    print(f"\n[6/6] Marking selected points and getting query point...")

    # Mark selected points
    selected_ids = {
        "topk": [r.chunk_id for r in results["topk"].results],
        "mmr": [r.chunk_id for r in results["mmr"].results],
        "qubo": [r.chunk_id for r in results["qubo"].results],
    }
    print(f"  Selected IDs: {len(selected_ids['topk'])} topk, {len(selected_ids['mmr'])} mmr, {len(selected_ids['qubo'])} qubo")

    umap_points = viz_service.mark_selected_points(umap_points, selected_ids)
    print("[OK] Points marked")

    # Get query point
    query_point = viz_service.get_query_point(query, dataset)
    print(f"[OK] Query point: {query_point}")

    print("\n" + "=" * 60)
    print("SUCCESS - ALL TESTS PASSED!")
    print("=" * 60)
    print("\nThe compare endpoint logic is working correctly.")
    print("If the API still returns 500, the issue may be:")
    print("  1. Request/response serialization")
    print("  2. Pydantic model validation")
    print("  3. CORS or middleware issues")

except Exception as e:
    print(f"\n[FAILED] TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
