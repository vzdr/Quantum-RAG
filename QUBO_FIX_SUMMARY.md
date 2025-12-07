# QUBO Implementation Fix - Summary

## Changes Made

### 1. Complete Rewrite of `core/qubo_solver.py`

**Simplified to core formula**: `Energy = alpha * s^T Q s - h^T s + P * (s^T 1 - k)^2`

**Key improvements**:
- Removed unnecessary class abstractions (QUBOProblem, etc.)
- Direct implementation of the mathematical formula
- Clear separation of concerns: similarity computation → QUBO matrix building → solving
- Better documentation explaining the formula at each step

**Critical bug fixes**:
1. **Off-diagonal penalty**: Changed from `2 * penalty` to `penalty / 2.0`
   - Reason: Symmetric matrix counts each pair twice in `s^T Q s`
   - Before: Each pair contributed `4P` (2P coefficient × 2 from symmetry)
   - After: Each pair contributes `P` (P/2 coefficient × 2 from symmetry) ✓

2. **Relevance term sign**: Kept as `-query_sims[i]` on diagonal
   - This is correct: we **minimize** energy, and want to **maximize** relevance
   - Negative coefficient means selecting high-similarity items reduces energy

**QUBO Matrix Structure** (k=5, P=1000):
```python
# Diagonal
Q[i, i] = -h[i] + P * (1 - 2k)
        = -query_sim[i] + 1000 * (1 - 10)
        = -query_sim[i] - 9000

# Off-diagonal
Q[i, j] = (alpha * pairwise_sim[i,j] + P) / 2
        = (0.05 * sim[i,j] + 1000) / 2
        ≈ 500 (when sim ≈ 0.5)
```

### 2. Updated `experiments/exp_1_1_poisoned_stress_test.py`

**Changed line 174**:
```python
# Before
strategy_qubo = create_retrieval_strategy('qubo', alpha=0, ...)

# After
strategy_qubo = create_retrieval_strategy('qubo', alpha=0.05, ...)
```

**Why**: `alpha=0` disabled diversity entirely, making QUBO equivalent to Top-K but with solver overhead

### 3. Updated `core/retrieval_strategies.py`

**Changed solver options handling**:
```python
# Before
self.solver_options = SOLVER_PRESETS[solver_preset].copy()

# After
self.solver_options = {'preset': solver_preset}
```

**Why**: The new `solve_diverse_retrieval_qubo` handles preset expansion internally

## Formula Verification

### Energy Function
```
E = alpha * s^T Q_pairwise s - h^T s + P * (s^T 1 - k)^2
```

Where:
- `s`: Binary selection vector {0,1}^n
- `Q_pairwise`: Chunk-to-chunk cosine similarity (n×n)
- `h`: Chunk-to-query cosine similarity (n,)
- `alpha`: Diversity weight (0.05 recommended)
- `P`: Cardinality penalty (1000 recommended)
- `k`: Number of selections (5 in experiments)

### Expansion
```
E = alpha * Σᵢⱼ Q_pairwise[i,j] * sᵢ * sⱼ
    - Σᵢ h[i] * sᵢ
    + P * (Σᵢ sᵢ - k)²

Cardinality term expansion:
(Σᵢ sᵢ - k)² = Σᵢ sᵢ² + Σᵢⱼ(i≠j) sᵢsⱼ - 2k Σᵢ sᵢ + k²
             = Σᵢ sᵢ + Σᵢⱼ(i≠j) sᵢsⱼ - 2k Σᵢ sᵢ + k²  (binary: sᵢ² = sᵢ)
             = Σᵢ (1-2k)sᵢ + Σᵢⱼ(i≠j) sᵢsⱼ + k²
```

### QUBO Matrix (s^T Q s form)
```
Q[i,i] = -h[i] + P(1 - 2k)           # Diagonal
Q[i,j] = (alpha * Q_pairwise[i,j] + P) / 2   # Off-diagonal (symmetric)
```

The `/2` is critical because `s^T Q s` expands to:
```
s^T Q s = Σᵢ Q[i,i]sᵢ² + Σᵢ<ⱼ Q[i,j]sᵢsⱼ + Σⱼ<ᵢ Q[i,j]sᵢsⱼ
        = Σᵢ Q[i,i]sᵢ + 2 Σᵢ<ⱼ Q[i,j]sᵢsⱼ
```

Each off-diagonal element is counted **twice**, hence we divide by 2.

## Expected Results

With the fixes applied:

### Energy Scale
For k=5 selections with typical similarities ~0.5-0.7:
- **Diagonal contribution**: `5 * (-0.6 - 9000) ≈ -45,003`
- **Off-diagonal contribution**: `10 pairs * 2 * 500 ≈ 10,000`
- **Total**: `~-35,000`
- **Similarity impact**: `~[-5, 0]` (now meaningful!)

### Aspect Recall (Expected)
- **Level 0** (no redundancy): QUBO ≈ 80-100% (similar to Top-K/MMR)
- **Level 1-2** (low redundancy): QUBO ≈ 90-100% (diversity helps)
- **Level 3-5** (high redundancy): QUBO ≈ 90-100% (diversity critical)
- **Top-K at Level 5**: Should degrade to <30% (baseline failure)
- **MMR**: Should be 70-85% (better than Top-K but not optimal)

### Key Success Metrics
- ✓ QUBO maintains >90% recall across ALL levels
- ✓ Top-K drops to <30% at level 5
- ✓ Lower intra-list similarity for QUBO vs Top-K
- ✓ Cardinality constraint satisfied (exactly k=5 selections)

## Testing

Run the experiment:
```bash
python experiments/exp_1_1_poisoned_stress_test.py --num-prompts 100
```

Check for:
1. QUBO recall consistently >90%
2. Solution quality metrics show diversity improvement
3. Energy values make sense (not dominated by penalty alone)
4. Exactly 5 chunks selected in all cases

## Verification Steps

If results still aren't good:

1. **Check QUBO energy components** in solution_quality metadata:
   - Is `avg_relevance` reasonable (0.5-0.8)?
   - Is `intra_list_similarity` lower than Top-K?
   - Is `constraint_violation` = 0?

2. **Verify matrix values**:
   ```python
   # Add debug print in build_qubo_matrix
   print(f"Diagonal range: [{np.min(np.diag(Q)):.2f}, {np.max(np.diag(Q)):.2f}]")
   print(f"Off-diag range: [{np.min(Q[Q!=np.diag(Q)]):.2f}, {np.max(Q[Q!=np.diag(Q)]):.2f}]")
   ```

3. **Test with brute-force solver** on small problem (n=15):
   ```python
   strategy_qubo = create_retrieval_strategy('qubo', alpha=0.05, solver='bruteforce')
   ```
   This guarantees optimal solution to verify formulation is correct.

## Code Simplification Benefits

1. **Clarity**: Formula is visible in code comments and structure
2. **Debuggability**: Each step can be inspected independently
3. **Maintainability**: No nested classes obscuring logic
4. **Correctness**: Direct translation from math to code reduces errors
5. **Performance**: Removed unnecessary abstractions

The rewrite reduces the solver module from ~666 lines to ~519 lines while adding MORE documentation.
