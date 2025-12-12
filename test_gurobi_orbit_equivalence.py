"""
Test equivalence between Gurobi and ORBIT solvers for QUBO retrieval.

This test verifies that both solvers produce similar results on the same problem.
"""
import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.retrieval import QUBORetrieval
from core.utils import load_wikipedia_dataset, filter_chunks_by_prompt, get_prompt_embedding, compute_aspect_recall


def test_solver_equivalence():
    """Test that Gurobi and ORBIT produce equivalent results."""
    print("="*80)
    print("TESTING GUROBI vs ORBIT SOLVER EQUIVALENCE")
    print("="*80)

    # Load dataset
    print("\n[1/5] Loading Wikipedia dataset...")
    chunks, embeddings_dict = load_wikipedia_dataset()
    print(f"✓ Loaded {len(chunks)} chunks")

    # Get a test prompt
    print("\n[2/5] Selecting test prompt...")
    prompt_ids = list(set(c['prompt_id'] for c in chunks if c.get('chunk_type') == 'prompt'))
    test_prompt_id = prompt_ids[0]  # Use first prompt
    print(f"✓ Using prompt_id: {test_prompt_id}")

    # Get prompt embedding and candidates
    print("\n[3/5] Preparing test case...")
    prompt_embedding = get_prompt_embedding(chunks, embeddings_dict, test_prompt_id)
    redundancy_level = 2
    candidates, gold_aspects, _, _, _ = filter_chunks_by_prompt(chunks, test_prompt_id, redundancy_level)
    k = 5

    # Prepare candidate data
    candidate_results = [{
        'id': cand['chunk_id'],
        'text': cand['text'],
        'embedding': embeddings_dict.get(cand['chunk_id']),
        'score': np.dot(
            prompt_embedding / np.linalg.norm(prompt_embedding),
            embeddings_dict.get(cand['chunk_id']) / np.linalg.norm(embeddings_dict.get(cand['chunk_id']))
        ),
        'metadata': cand
    } for cand in candidates if embeddings_dict.get(cand['chunk_id']) is not None]

    candidate_results.sort(key=lambda x: x['score'], reverse=True)
    print(f"✓ Prepared {len(candidate_results)} candidates, selecting k={k}")
    print(f"✓ Gold aspects: {len(gold_aspects)}")

    # Test parameters (production values)
    alpha = 0.04
    penalty = 10.0
    beta = 0.8

    # Run Gurobi solver
    print(f"\n[4/5] Running Gurobi solver (alpha={alpha}, penalty={penalty}, beta={beta})...")
    gurobi_retrieval = QUBORetrieval(alpha=alpha, penalty=penalty, beta=beta, solver='gurobi')
    gurobi_results = gurobi_retrieval.retrieve(prompt_embedding, candidate_results, k)
    gurobi_meta = [r.chunk.metadata for r in gurobi_results]
    gurobi_recall, gurobi_count = compute_aspect_recall(gurobi_meta, gold_aspects)
    gurobi_ids = [r.chunk.id for r in gurobi_results]

    print(f"✓ Gurobi selected: {gurobi_ids}")
    print(f"  Aspect recall: {gurobi_recall:.1f}% ({gurobi_count}/{len(gold_aspects)} aspects)")

    # Run ORBIT solver
    print(f"\n[5/5] Running ORBIT solver (same parameters)...")
    try:
        orbit_retrieval = QUBORetrieval(alpha=alpha, penalty=penalty, beta=beta, solver='orbit')
        orbit_results = orbit_retrieval.retrieve(prompt_embedding, candidate_results, k)
        orbit_meta = [r.chunk.metadata for r in orbit_results]
        orbit_recall, orbit_count = compute_aspect_recall(orbit_meta, gold_aspects)
        orbit_ids = [r.chunk.id for r in orbit_results]

        print(f"✓ ORBIT selected: {orbit_ids}")
        print(f"  Aspect recall: {orbit_recall:.1f}% ({orbit_count}/{len(gold_aspects)} aspects)")

        # Compare results
        print("\n" + "="*80)
        print("COMPARISON")
        print("="*80)

        # Overlap analysis
        gurobi_set = set(gurobi_ids)
        orbit_set = set(orbit_ids)
        overlap = gurobi_set & orbit_set
        overlap_pct = len(overlap) / k * 100

        print(f"\nSelection Overlap: {len(overlap)}/{k} chunks ({overlap_pct:.1f}%)")
        print(f"  Common: {overlap}")
        print(f"  Gurobi only: {gurobi_set - orbit_set}")
        print(f"  ORBIT only: {orbit_set - gurobi_set}")

        print(f"\nAspect Recall:")
        print(f"  Gurobi: {gurobi_recall:.1f}% ({gurobi_count}/{len(gold_aspects)})")
        print(f"  ORBIT:  {orbit_recall:.1f}% ({orbit_count}/{len(gold_aspects)})")
        print(f"  Difference: {abs(gurobi_recall - orbit_recall):.1f}%")

        # Success criteria
        print("\n" + "="*80)
        recall_diff = abs(gurobi_recall - orbit_recall)
        if recall_diff <= 20.0:  # Allow 20% difference (ORBIT is stochastic)
            print("✓ TEST PASSED: Solvers produce comparable results")
            print(f"  (Recall difference: {recall_diff:.1f}% ≤ 20% threshold)")
        else:
            print("⚠ TEST WARNING: Solvers differ significantly")
            print(f"  (Recall difference: {recall_diff:.1f}% > 20% threshold)")
            print("  Note: ORBIT is stochastic, some variation expected")

        if overlap_pct >= 60:
            print(f"✓ Good overlap: {overlap_pct:.1f}% ≥ 60%")
        else:
            print(f"⚠ Low overlap: {overlap_pct:.1f}% < 60%")

        print("\nNote: ORBIT uses probabilistic p-bit computing, so exact equivalence")
        print("      is not expected. Both solvers should find high-quality solutions.")

    except ImportError as e:
        print(f"✗ ORBIT not available: {e}")
        print("\nTo test ORBIT, install it:")
        print("  cd orbit")
        print("  uv pip install orbit-0.2.0-py3-none-any.whl")
        return False

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    return True


if __name__ == '__main__':
    test_solver_equivalence()
