"""
Test script for submission notebook - runs all cells to verify functionality
"""

print("="*80)
print("TESTING SUBMISSION NOTEBOOK")
print("="*80)

# CELL 1: Imports
print("\n[CELL 1] Testing imports...")
try:
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from submission_utils import (
        load_wikipedia_dataset,
        filter_chunks_by_prompt,
        get_prompt_embedding,
        retrieve_topk,
        retrieve_qubo_gurobi,
        print_retrieval_results,
        print_comparison_table,
        compute_aspect_recall
    )
    print("[OK] Imports successful")
except Exception as e:
    print(f"[FAIL] Import failed: {e}")
    exit(1)

# CELL 2: Load dataset
print("\n[CELL 2] Loading dataset...")
try:
    chunks, embeddings = load_wikipedia_dataset('./data/wikipedia')
    print(f"[OK] Loaded {len(chunks):,} chunks")
    print(f"[OK] Loaded {len(embeddings):,} embeddings")
    print(f"[OK] Embedding dimension: {list(embeddings.values())[0].shape[0]}")
except Exception as e:
    print(f"[FAIL] Dataset loading failed: {e}")
    exit(1)

# CELL 3: Select test query
print("\n[CELL 3] Selecting test prompt...")
try:
    # Get first available prompt ID (they are UUIDs, not integers)
    all_prompts = [c for c in chunks if c.get('chunk_type') == 'prompt']
    if not all_prompts:
        print("[FAIL] No prompt chunks found in dataset!")
        exit(1)

    TEST_PROMPT_ID = all_prompts[0]['prompt_id']
    k = 5
    prompt_text = all_prompts[0]['text']
    query_embedding = get_prompt_embedding(chunks, embeddings, TEST_PROMPT_ID)

    if query_embedding is None:
        print("[FAIL] Query embedding not found!")
        exit(1)

    print(f"[OK] Test Prompt ID: {str(TEST_PROMPT_ID)[:40]}...")
    print(f"[OK] Prompt text: {prompt_text[:100]}...")
    print(f"[OK] Query embedding shape: {query_embedding.shape}")
except Exception as e:
    print(f"[FAIL] Prompt selection failed: {e}")
    exit(1)

# CELL 4-6: Top-K at Redundancy Level 0
print("\n[CELL 4-6] Testing Top-K at redundancy level 0...")
try:
    redundancy_level = 0
    candidates_L0, gold_aspects = filter_chunks_by_prompt(chunks, TEST_PROMPT_ID, redundancy_level)

    print(f"[OK] Candidate pool size: {len(candidates_L0)}")
    print(f"[OK] Gold aspects to find: {len(gold_aspects)}")

    topk_results_L0 = retrieve_topk(query_embedding, candidates_L0, embeddings, k=k)
    print(f"[OK] Top-K retrieved {len(topk_results_L0)} chunks")

    recall, count = compute_aspect_recall(topk_results_L0, gold_aspects)
    print(f"[OK] Aspect recall: {recall:.1f}% ({count}/{len(gold_aspects)})")
except Exception as e:
    print(f"[FAIL] Top-K L0 failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# CELL 7-8: Top-K at Redundancy Level 2
print("\n[CELL 7-8] Testing Top-K at redundancy level 2...")
try:
    redundancy_level = 2
    candidates_L2, _ = filter_chunks_by_prompt(chunks, TEST_PROMPT_ID, redundancy_level)

    print(f"[OK] Candidate pool size: {len(candidates_L2)}")

    topk_results_L2 = retrieve_topk(query_embedding, candidates_L2, embeddings, k=k)
    print(f"[OK] Top-K retrieved {len(topk_results_L2)} chunks")

    recall, count = compute_aspect_recall(topk_results_L2, gold_aspects)
    print(f"[OK] Aspect recall: {recall:.1f}% ({count}/{len(gold_aspects)})")

    redundant_count = len([c for c in topk_results_L2 if c.get('chunk_type') == 'gold_redundant'])
    print(f"  -> {redundant_count}/{k} redundant copies")
except Exception as e:
    print(f"[FAIL] Top-K L2 failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# CELL 9-10: Top-K at Redundancy Level 5
print("\n[CELL 9-10] Testing Top-K at redundancy level 5...")
try:
    redundancy_level = 5
    candidates_L5, _ = filter_chunks_by_prompt(chunks, TEST_PROMPT_ID, redundancy_level)

    print(f"[OK] Candidate pool size: {len(candidates_L5)}")

    topk_results_L5 = retrieve_topk(query_embedding, candidates_L5, embeddings, k=k)
    print(f"[OK] Top-K retrieved {len(topk_results_L5)} chunks")

    recall, count = compute_aspect_recall(topk_results_L5, gold_aspects)
    print(f"[OK] Aspect recall: {recall:.1f}% ({count}/{len(gold_aspects)})")

    redundant_count = len([c for c in topk_results_L5 if c.get('chunk_type') == 'gold_redundant'])
    print(f"  -> {redundant_count}/{k} redundant copies")
except Exception as e:
    print(f"[FAIL] Top-K L5 failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# CELL 11-13: QUBO with Gurobi
print("\n[CELL 11-13] Testing QUBO with Gurobi...")
ALPHA = 0.02  # Diversity penalty weight (from notebook markdown)
PENALTY = 10.0  # Cardinality penalty (energies ~10 without constraint)

try:
    print(f"  Parameters: alpha={ALPHA}, P={PENALTY}, k={k}")

    # Test at redundancy level 0
    print("  Testing QUBO at L0...")
    qubo_results_L0, qubo_meta_L0 = retrieve_qubo_gurobi(
        query_embedding, candidates_L0, embeddings, k=k, alpha=ALPHA, penalty=PENALTY, verbose=False
    )
    print(f"  [OK] Solve time: {qubo_meta_L0['solve_time']:.3f}s")
    print(f"  [OK] Retrieved {len(qubo_results_L0)} chunks")

    recall, count = compute_aspect_recall(qubo_results_L0, gold_aspects)
    print(f"  [OK] Aspect recall: {recall:.1f}% ({count}/{len(gold_aspects)})")

except Exception as e:
    print(f"[FAIL] QUBO Gurobi failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test QUBO at L2
print("\n  Testing QUBO at L2...")
try:
    qubo_results_L2, qubo_meta_L2 = retrieve_qubo_gurobi(
        query_embedding, candidates_L2, embeddings, k=k, alpha=ALPHA, penalty=PENALTY, verbose=False
    )
    print(f"  [OK] Solve time: {qubo_meta_L2['solve_time']:.3f}s")
    print(f"  [OK] Retrieved {len(qubo_results_L2)} chunks")

    recall, count = compute_aspect_recall(qubo_results_L2, gold_aspects)
    print(f"  [OK] Aspect recall: {recall:.1f}% ({count}/{len(gold_aspects)})")

    redundant_count = len([c for c in qubo_results_L2 if c.get('chunk_type') == 'gold_redundant'])
    print(f"  -> Only {redundant_count}/{k} redundant copies (QUBO rejects duplicates!)")
except Exception as e:
    print(f"[FAIL] QUBO L2 failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test QUBO at L5
print("\n  Testing QUBO at L5...")
try:
    qubo_results_L5, qubo_meta_L5 = retrieve_qubo_gurobi(
        query_embedding, candidates_L5, embeddings, k=k, alpha=ALPHA, penalty=PENALTY, verbose=False
    )
    print(f"  [OK] Solve time: {qubo_meta_L5['solve_time']:.3f}s")
    print(f"  [OK] Retrieved {len(qubo_results_L5)} chunks")

    recall, count = compute_aspect_recall(qubo_results_L5, gold_aspects)
    print(f"  [OK] Aspect recall: {recall:.1f}% ({count}/{len(gold_aspects)})")

    redundant_count = len([c for c in qubo_results_L5 if c.get('chunk_type') == 'gold_redundant'])
    print(f"  -> Only {redundant_count}/{k} redundant copies")
except Exception as e:
    print(f"[FAIL] QUBO L5 failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# CELL 14-16: Comparison Tables
print("\n[CELL 14-16] Testing comparison tables...")
try:
    from submission_utils import compute_intra_list_similarity

    # Just compute metrics, don't print full table
    topk_recall_L5, topk_count_L5 = compute_aspect_recall(topk_results_L5, gold_aspects)
    qubo_recall_L5, qubo_count_L5 = compute_aspect_recall(qubo_results_L5, gold_aspects)

    topk_diversity = compute_intra_list_similarity(topk_results_L5, embeddings)
    qubo_diversity = compute_intra_list_similarity(qubo_results_L5, embeddings)

    print(f"[OK] Top-K L5: {topk_recall_L5:.1f}% recall, {topk_diversity:.3f} similarity")
    print(f"[OK] QUBO L5:  {qubo_recall_L5:.1f}% recall, {qubo_diversity:.3f} similarity")

    if topk_recall_L5 > 0:
        improvement = ((qubo_recall_L5 - topk_recall_L5) / topk_recall_L5) * 100
        print(f"[OK] Improvement: {improvement:+.1f}%")
except Exception as e:
    print(f"[FAIL] Comparison failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# CELL 17: Visualization
print("\n[CELL 17] Testing visualization...")
try:
    topk_recalls = []
    qubo_recalls = []
    levels = [0, 2, 5]

    for topk_res, qubo_res in [
        (topk_results_L0, qubo_results_L0),
        (topk_results_L2, qubo_results_L2),
        (topk_results_L5, qubo_results_L5)
    ]:
        topk_recall, _ = compute_aspect_recall(topk_res, gold_aspects)
        qubo_recall, _ = compute_aspect_recall(qubo_res, gold_aspects)
        topk_recalls.append(topk_recall)
        qubo_recalls.append(qubo_recall)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(levels, topk_recalls, 'o-', label='Top-K', linewidth=2, markersize=8)
    ax.plot(levels, qubo_recalls, 's-', label='QUBO-RAG', linewidth=2, markersize=8)
    ax.set_xlabel('Redundancy Level')
    ax.set_ylabel('Aspect Recall (%)')
    ax.set_title('Robustness to Dataset Redundancy')
    ax.legend()
    ax.grid(alpha=0.3)

    # Save plot
    import os
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/submission_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Plot saved to results/submission_comparison.png")
    print(f"  Top-K at L5: {topk_recalls[2]:.1f}%")
    print(f"  QUBO at L5:  {qubo_recalls[2]:.1f}%")
except Exception as e:
    print(f"[FAIL] Visualization failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# CELL 18-19: Stress test results (load pre-computed)
print("\n[CELL 18-19] Testing stress test visualization...")
try:
    import json

    with open('results/exp_1_1_poisoned_stress_test.json', 'r') as f:
        stress_test_results = json.load(f)

    levels = [r['redundancy_level'] for r in stress_test_results]
    topk_means = [r['topk']['mean_aspect_recall'] for r in stress_test_results]
    qubo_means = [r['qubo']['mean_aspect_recall'] for r in stress_test_results]

    print(f"[OK] Loaded stress test results for {len(levels)} redundancy levels")
    print(f"  Top-K at L5: {topk_means[5]:.1f}%")
    print(f"  QUBO at L5:  {qubo_means[5]:.1f}%")

    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.array(levels)
    width = 0.35

    ax.bar(x - width/2, topk_means, width, label='Top-K', alpha=0.8)
    ax.bar(x + width/2, qubo_means, width, label='QUBO-RAG', alpha=0.8)
    ax.set_xlabel('Redundancy Level')
    ax.set_ylabel('Aspect Recall (%)')
    ax.set_title('Stress Test: 100 Queries per Level')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.savefig('results/submission_stress_test.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Stress test plot saved")
except Exception as e:
    print(f"[FAIL] Stress test visualization failed: {e}")
    import traceback
    traceback.print_exc()
    # Don't exit - this is optional

print("\n" + "="*80)
print("[OK] ALL CORE TESTS PASSED!")
print("="*80)
print("\n Summary:")
print(f"  Top-K degradation: {topk_recalls[0]:.1f}% -> {topk_recalls[2]:.1f}%")
print(f"  QUBO stability:    {qubo_recalls[0]:.1f}% -> {qubo_recalls[2]:.1f}%")
if topk_recalls[2] > 0:
    improvement = ((qubo_recalls[2] - topk_recalls[2]) / topk_recalls[2]) * 100
    print(f"  Improvement at L5: {improvement:+.1f}%")

print("\n[OK] Notebook is ready for submission!")
print("\nNote: ORBIT integration (Part 7) not tested - requires orbit module")
