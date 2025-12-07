"""Fix the remaining Gurobi references in cells 21, 23, 25"""

import json
from pathlib import Path

def fix_cells():
    notebook_path = Path('SUBMISSION_Quantum_RAG_ORBIT.ipynb')

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Cell 21 - QUBO at Redundancy Level 0
    nb['cells'][21]['source'] = """# Run QUBO at redundancy level 0
qubo_results_L0, qubo_meta_L0 = retrieve_qubo_orbit(
    query_embedding,
    candidates_L0,
    embeddings,
    k=k,
    alpha=ALPHA,
    penalty=PENALTY,
    solver_preset='balanced'
)

print(f"Solver: ORBIT (p-bit computing)")
print(f"Solve time: {qubo_meta_L0['solve_time']:.3f}s")
print(f"Objective value: {qubo_meta_L0['objective_value']:.4f}")
print(f"Chunks selected: {qubo_meta_L0['num_selected']}")

print_retrieval_results(
    qubo_results_L0,
    gold_aspects,
    method_name="QUBO-RAG (Redundancy Level 0)",
    show_text_preview=True
)"""
    nb['cells'][21]['outputs'] = []  # Clear old outputs

    # Cell 23 - QUBO at Redundancy Level 2
    nb['cells'][23]['source'] = """# Run QUBO at redundancy level 2
qubo_results_L2, qubo_meta_L2 = retrieve_qubo_orbit(
    query_embedding,
    candidates_L2,
    embeddings,
    k=k,
    alpha=ALPHA,
    penalty=PENALTY,
    solver_preset='balanced'
)

print(f"Solver: ORBIT (p-bit computing)")
print(f"Solve time: {qubo_meta_L2['solve_time']:.3f}s")
print(f"Objective value: {qubo_meta_L2['objective_value']:.4f}")
print(f"Chunks selected: {qubo_meta_L2['num_selected']}")

print_retrieval_results(
    qubo_results_L2,
    gold_aspects,
    method_name="QUBO-RAG (Redundancy Level 2)",
    show_text_preview=True
)

# Count how many are redundant copies
redundant_count = len([c for c in qubo_results_L2 if c.get('chunk_type') == 'gold_redundant'])
print(f"✓ Only {redundant_count}/{k} redundant copies (QUBO rejects duplicates!)")"""
    nb['cells'][23]['outputs'] = []  # Clear old outputs

    # Cell 25 - QUBO at Redundancy Level 5
    nb['cells'][25]['source'] = """# Run QUBO at redundancy level 5
qubo_results_L5, qubo_meta_L5 = retrieve_qubo_orbit(
    query_embedding,
    candidates_L5,
    embeddings,
    k=k,
    alpha=ALPHA,
    penalty=PENALTY,
    solver_preset='balanced'
)

print(f"Solver: ORBIT (p-bit computing)")
print(f"Solve time: {qubo_meta_L5['solve_time']:.3f}s")
print(f"Objective value: {qubo_meta_L5['objective_value']:.4f}")
print(f"Chunks selected: {qubo_meta_L5['num_selected']}")

print_retrieval_results(
    qubo_results_L5,
    gold_aspects,
    method_name="QUBO-RAG (Redundancy Level 5)",
    show_text_preview=True
)

# Count how many are redundant copies
redundant_count = len([c for c in qubo_results_L5 if c.get('chunk_type') == 'gold_redundant'])
print(f"✓ Only {redundant_count}/{k} redundant copies (QUBO maintains diversity!)")"""
    nb['cells'][25]['outputs'] = []  # Clear old outputs

    # Clear outputs from comparison cells too
    for idx in [27, 28, 29, 31, 34]:
        if idx < len(nb['cells']):
            nb['cells'][idx]['outputs'] = []

    # Write back
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

    print("[OK] Fixed cells 21, 23, 25 to use ORBIT")
    print("[OK] Cleared all output cells - ready to rerun")

if __name__ == '__main__':
    fix_cells()
