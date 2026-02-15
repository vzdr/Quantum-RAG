"""
Test script to compare ORBIT and Gurobi solvers on the same QUBO problem.
This will help us identify why they're giving different results.
"""

import numpy as np
from core.qubo_solver import (
    solve_diverse_retrieval_qubo,
    compute_cosine_similarities,
    evaluate_energy
)
from submission_utils import (
    load_wikipedia_dataset,
    filter_chunks_by_prompt,
    get_prompt_embedding
)

def main():
    print("="*80)
    print("SOLVER COMPARISON TEST")
    print("="*80)

    # Load data
    print("\nLoading dataset...")
    chunks, embeddings = load_wikipedia_dataset('./data/wikipedia')

    # Get first prompt
    all_prompts = [c for c in chunks if c.get('chunk_type') == 'prompt']
    TEST_PROMPT_ID = all_prompts[0]['prompt_id']
    query_embedding = get_prompt_embedding(chunks, embeddings, TEST_PROMPT_ID)

    # Test at redundancy level 5 (where we saw the discrepancy)
    print(f"\nTesting at Redundancy Level 5...")
    candidates_L5, gold_aspects = filter_chunks_by_prompt(chunks, TEST_PROMPT_ID, 5)

    # Prepare embeddings array
    chunk_embs = []
    for chunk in candidates_L5:
        chunk_id = chunk['chunk_id']
        chunk_emb = embeddings.get(chunk_id)
        if chunk_emb is not None:
            chunk_embs.append(chunk_emb)

    candidate_embs_array = np.array(chunk_embs)

    print(f"Number of candidates: {len(candidate_embs_array)}")
    print(f"Gold aspects: {len(gold_aspects)}")

    # Test parameters
    k = 5
    test_alphas = [0.02, 0.05, 0.10, 0.20, 0.50]
    penalty = 10.0

    print(f"\nTesting different alpha values:")
    print(f"k = {k}, penalty = {penalty}")
    print()

    for alpha in test_alphas:
        print(f"\n{'='*80}")
        print(f"ALPHA = {alpha}")
        print(f"{'='*80}")

        # Solve with Gurobi
        print("\n1. GUROBI:")
        try:
            gurobi_indices, gurobi_meta = solve_diverse_retrieval_qubo(
                query_embedding=query_embedding,
                candidate_embeddings=candidate_embs_array,
                k=k,
                alpha=alpha,
                penalty=penalty,
                solver='gurobi',
                solver_options={'OutputFlag': 0}
            )

            # Count unique aspects
            gurobi_aspects = set()
            for idx in gurobi_indices:
                aspect_id = candidates_L5[idx].get('aspect_id', -1)
                if aspect_id >= 0:
                    gurobi_aspects.add(aspect_id)

            gurobi_recall = len(gurobi_aspects & gold_aspects) / len(gold_aspects) * 100

            print(f"  Energy: {gurobi_meta['energy']:.4f}")
            print(f"  Selected: {gurobi_meta['solution_quality']['n_selected']}")
            print(f"  Aspect Recall: {gurobi_recall:.1f}% ({len(gurobi_aspects & gold_aspects)}/{len(gold_aspects)})")
            print(f"  Avg Relevance: {gurobi_meta['solution_quality']['avg_relevance']:.4f}")
            print(f"  Intra-List Sim: {gurobi_meta['solution_quality']['intra_list_similarity']:.4f}")
            print(f"  Time: {gurobi_meta['execution_time']:.3f}s")

        except Exception as e:
            print(f"  ERROR: {e}")
            gurobi_indices = None
            gurobi_recall = 0

        # Solve with ORBIT
        print("\n2. ORBIT:")
        try:
            orbit_indices, orbit_meta = solve_diverse_retrieval_qubo(
                query_embedding=query_embedding,
                candidate_embeddings=candidate_embs_array,
                k=k,
                alpha=alpha,
                penalty=penalty,
                solver='orbit',
                solver_options={'preset': 'balanced'}
            )

            # Count unique aspects
            orbit_aspects = set()
            for idx in orbit_indices:
                aspect_id = candidates_L5[idx].get('aspect_id', -1)
                if aspect_id >= 0:
                    orbit_aspects.add(aspect_id)

            orbit_recall = len(orbit_aspects & gold_aspects) / len(gold_aspects) * 100

            print(f"  Energy: {orbit_meta['energy']:.4f}")
            print(f"  Selected: {orbit_meta['solution_quality']['n_selected']}")
            print(f"  Aspect Recall: {orbit_recall:.1f}% ({len(orbit_aspects & gold_aspects)}/{len(gold_aspects)})")
            print(f"  Avg Relevance: {orbit_meta['solution_quality']['avg_relevance']:.4f}")
            print(f"  Intra-List Sim: {orbit_meta['solution_quality']['intra_list_similarity']:.4f}")
            print(f"  Time: {orbit_meta['execution_time']:.3f}s")

        except Exception as e:
            print(f"  ERROR: {e}")
            orbit_indices = None
            orbit_recall = 0

        # Compare
        print("\n3. COMPARISON:")
        if gurobi_indices is not None and orbit_indices is not None:
            print(f"  Recall difference: {abs(gurobi_recall - orbit_recall):.1f}%")
            print(f"  Energy difference: {abs(gurobi_meta['energy'] - orbit_meta['energy']):.4f}")

            # Check if same indices selected
            if set(gurobi_indices) == set(orbit_indices):
                print(f"  ✓ Same indices selected")
            else:
                print(f"  ✗ Different indices selected")
                print(f"    Gurobi only: {set(gurobi_indices) - set(orbit_indices)}")
                print(f"    ORBIT only: {set(orbit_indices) - set(gurobi_indices)}")

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
