# QUBO Implementation Bug Analysis

## Critical Bug Found

**Location**: `experiments/exp_1_1_poisoned_stress_test.py:174`

```python
strategy_qubo = create_retrieval_strategy('qubo', alpha=0, penalty=1000.0,
                                           solver='orbit', solver_preset='balanced')
```

## Problem

The QUBO strategy is being initialized with **`alpha=0`**, which completely disables the diversity mechanism.

### QUBO Formulation Review

From `core/qubo_solver.py`, the QUBO matrix is built as:

```python
# Diagonal (linear terms)
Q[i, i] = -1 * query_sims[i] + penalty * (1 - 2*k)

# Off-diagonal (quadratic terms)
Q[i, j] = alpha * pairwise_sims[i, j] + 2 * penalty
```

The objective function being minimized is:
```
f(x) = -Σᵢ sim(q, cᵢ) * xᵢ + α * Σᵢⱼ sim(cᵢ, cⱼ) * xᵢ * xⱼ + P * (Σxᵢ - k)²
       └──────relevance──────┘   └────────diversity penalty────────┘   └─constraint─┘
```

### Impact of alpha=0

When `alpha=0`:
- **Diversity term is completely removed**: `α * Σᵢⱼ sim(cᵢ, cⱼ) * xᵢ * xⱼ = 0`
- The QUBO reduces to: `f(x) = -Σᵢ sim(q, cᵢ) * xᵢ + P * (Σxᵢ - k)²`
- This is **equivalent to Top-K selection** with only a cardinality constraint
- There is **no penalty for selecting similar documents**
- The solver has **no incentive to prefer diverse results**

## Why QUBO is Performing Poorly

### Experiment Results with alpha=0:
- Level 0: QUBO = 20% (Top-K = 80%)
- Level 1: QUBO = 20% (Top-K = 80%)
- Level 2: QUBO = 60% (Top-K = 80%)
- Level 3: QUBO = 80% (Top-K = 40%)
- Level 5: QUBO = 60% (Top-K = 60%)

With `alpha=0`, QUBO is actually performing **worse than Top-K** at low redundancy levels!

This makes sense because:
1. At low redundancy (levels 0-1), Top-K naturally gets diverse results since there's little redundancy
2. QUBO with alpha=0 has the **overhead of quantum solving** but provides **no diversity benefit**
3. The ORBIT solver may even introduce noise/randomness that makes it worse than deterministic Top-K
4. At higher redundancy levels (3-5), QUBO sometimes performs better due to random chance, not design

## Correct Configuration

Based on the codebase documentation and standard practice:

```python
strategy_qubo = create_retrieval_strategy('qubo', alpha=0.05, penalty=1000.0,
                                           solver='orbit', solver_preset='balanced')
```

### Why alpha=0.05?

From `core/retrieval_strategies.py` line 305:
```python
alpha: Diversity weight (default: 0.05). Controls penalty for similar documents.
```

The value 0.05 provides a good balance:
- **Not too small** (like 0): Would ignore diversity
- **Not too large** (like 0.5): Would over-penalize similarity and hurt relevance
- **Moderate weight**: Balances relevance with diversity

### Expected Behavior with alpha=0.05

The QUBO formulation should:
1. **Reward relevance**: High `sim(q, cᵢ)` increases selection probability
2. **Penalize redundancy**: High `sim(cᵢ, cⱼ)` for selected pairs decreases objective
3. **Enforce cardinality**: Penalty term ensures exactly k selections
4. **Achieve better aspect recall**: Especially as redundancy increases

## Verification Needed

After fixing alpha to 0.05, we should verify:

1. **QUBO energy components**: Check `solution_quality` metadata to ensure diversity term is non-zero
2. **Aspect recall improvement**: QUBO should maintain >90% across all redundancy levels
3. **Intra-list similarity**: Should be lower than Top-K/MMR, indicating more diverse selections
4. **Constraint satisfaction**: Solver should still select exactly k=5 chunks

## Additional Observations

### Solver Implementation (Appears Correct)

The QUBO solver implementation in `core/qubo_solver.py` looks correct:
- Matrix construction follows standard QUBO formulation
- Ising conversion is mathematically accurate
- ORBIT integration is properly configured
- Cardinality constraint enforcement via penalty method is appropriate

### Potential Secondary Issues

Even with correct alpha, if QUBO still underperforms, investigate:
1. **Penalty strength**: `penalty=1000.0` may be too weak or too strong
2. **ORBIT solver settings**: `balanced` preset may need tuning
3. **Similarity normalization**: Ensure embeddings are properly normalized
4. **Energy scale**: Check if relevance and diversity terms are on similar scales

## Recommended Fix

```python
# experiments/exp_1_1_poisoned_stress_test.py, line 174
strategy_qubo = create_retrieval_strategy('qubo', alpha=0.05, penalty=1000.0,
                                           solver='orbit', solver_preset='balanced')
```

This single-line change should dramatically improve QUBO performance.
