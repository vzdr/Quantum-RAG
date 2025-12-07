# QUBO Cardinality Constraint Bug Analysis

## The Problem

The cardinality constraint in the QUBO formulation is **incorrectly implemented**, causing the penalty term to dominate and produce nonsensical results.

## Mathematical Background

### Cardinality Constraint Expansion

We want to enforce `Σxᵢ = k` by minimizing the penalty term:
```
P * (Σxᵢ - k)²
```

Expanding this:
```
(Σxᵢ - k)² = (Σᵢ xᵢ)² - 2k(Σᵢ xᵢ) + k²

= Σᵢ xᵢ² + Σᵢ Σⱼ≠ᵢ xᵢxⱼ - 2k Σᵢ xᵢ + k²
```

For binary variables where `xᵢ ∈ {0,1}`, we have `xᵢ² = xᵢ`, so:
```
= Σᵢ xᵢ + Σᵢ Σⱼ≠ᵢ xᵢxⱼ - 2k Σᵢ xᵢ + k²

= Σᵢ (1 - 2k)xᵢ + Σᵢ Σⱼ≠ᵢ xᵢxⱼ + k²
```

The constant `k²` doesn't affect optimization, so we get:
```
P * (Σxᵢ - k)² ≈ P * [Σᵢ (1 - 2k)xᵢ + Σᵢ Σⱼ≠ᵢ xᵢxⱼ]
```

This gives coefficients:
- **Linear term**: `P(1 - 2k)` for each `xᵢ`
- **Quadratic term**: `P` for each pair `xᵢxⱼ` where `i ≠ j`

## QUBO Matrix Representation

A QUBO problem is represented as:
```
minimize: E(x) = Σᵢⱼ Q[i,j] * xᵢ * xⱼ = x^T Q x
```

For a symmetric matrix Q:
- **Diagonal elements** `Q[i,i]`: Coefficient for `xᵢ²`
- **Off-diagonal elements** `Q[i,j]` where `i≠j`: Coefficient for `xᵢxⱼ`

### Critical Point: Symmetric Matrix Double-Counting

When we compute `x^T Q x` with a symmetric matrix:
```
E = Σᵢ Q[i,i]xᵢ² + Σᵢ Σⱼ≠ᵢ Q[i,j]xᵢxⱼ
```

For off-diagonal terms where `Q[i,j] = Q[j,i]`:
```
Σᵢ Σⱼ≠ᵢ Q[i,j]xᵢxⱼ = Σᵢ<ⱼ Q[i,j]xᵢxⱼ + Σⱼ<ᵢ Q[i,j]xᵢxⱼ
                     = Σᵢ<ⱼ Q[i,j]xᵢxⱼ + Σᵢ<ⱼ Q[j,i]xⱼxᵢ
                     = 2 * Σᵢ<ⱼ Q[i,j]xᵢxⱼ    (since Q is symmetric)
```

Therefore, if we want a coefficient of `P` for each unique pair `(i,j)` where `i<j`, we must set:
```
Q[i,j] = P/2    (for i < j)
Q[j,i] = P/2    (for symmetry)
```

Then the contribution to energy is: `2 * (P/2) * xᵢxⱼ = P * xᵢxⱼ` ✓

## Bug in Current Implementation

From `core/qubo_solver.py` lines 86-96:

```python
# Build off-diagonal (quadratic terms + cardinality penalty)
for i in range(self.n):
    for j in range(i + 1, self.n):
        # Diversity term (weighted by alpha)
        Q[i, j] = self.alpha * self.pairwise_sims[i, j]

        # Cardinality penalty: 2P
        Q[i, j] += 2 * self.penalty  # ❌ BUG: Should be just self.penalty

        # Ensure symmetry
        Q[j, i] = Q[i, j]
```

### What Happens

The code sets `Q[i,j] = 2P` (plus diversity term).

When we compute `x^T Q x`:
```
Each pair (i,j) contributes: 2 * 2P * xᵢxⱼ = 4P * xᵢxⱼ
```

This is **4 times too large**!

### Impact with k=5, P=1000

With the bug:
- Diagonal: `Q[i,i] = -sim[i] + 1000(1 - 10) = -sim[i] - 9000`
- Off-diagonal: `Q[i,j] = 0 + 2*1000 = 2000`
- Each pair contributes: `4000` to the energy

The cardinality penalty completely dominates:
- Similarity scores are in `[0, 1]`
- Diagonal from similarity: `~[-1, 0]`
- Diagonal from penalty: `-9000`
- Off-diagonal from penalty: `+4000` per pair

For any selection of k=5 items:
- Diagonal contribution: `5 * (-9000) = -45000` (roughly constant)
- Off-diagonal contribution: `C(5,2) * 4000 = 10 * 4000 = 40000`
- Total penalty contribution: `~-5000`
- Similarity contribution: `~[-5, 0]`

The penalty completely dominates (1000x larger than similarity)!

## Why Results Are Random/Poor

With such massive penalty terms:
1. **Similarity becomes irrelevant**: The -9000 to +4000 penalty terms dwarf the ±1 similarity
2. **ORBIT solver struggles**: The energy landscape is dominated by random noise from numerical precision
3. **Cardinality is over-enforced**: Any deviation from exactly k selections is catastrophically punished
4. **No optimization signal**: The actual objective (relevance + diversity) is lost in the noise

This explains why QUBO gets 20% recall at level 0 where Top-K gets 80%:
- Top-K correctly ranks by similarity
- QUBO's ranking is essentially random noise because similarity signal is buried

## Correct Implementation

```python
# Build off-diagonal (quadratic terms + cardinality penalty)
for i in range(self.n):
    for j in range(i + 1, self.n):
        # Diversity term (weighted by alpha)
        Q[i, j] = self.alpha * self.pairwise_sims[i, j]

        # Cardinality penalty: P (NOT 2P!)
        Q[i, j] += self.penalty  # ✓ FIXED

        # Ensure symmetry
        Q[j, i] = Q[i, j]
```

## Alternative Formulations

### Option 1: Upper Triangle Only (Current Approach - Fixed)
```python
Q[i, j] = alpha * sim[i,j] + P    # for i < j
Q[j, i] = Q[i, j]                  # symmetric
```
Energy: `x^T Q x` automatically doubles off-diagonal terms

### Option 2: Explicit Doubling in Formula
```python
# Only fill upper triangle, no symmetry copy
Q[i, j] = 2 * (alpha * sim[i,j] + P)    # for i < j
# Don't copy to Q[j,i]
```
Energy: Need special handling to only sum upper triangle

The current code tries to use Option 1 but incorrectly doubles the penalty.

## Verification

After fixing, verify:
1. **Energy scale**: Similarity terms should not be dwarfed by penalty
2. **Typical diagonal value**: Should be close to `-sim[i] + P(1-2k)` ~ `-0.7 - 9000` = `-9000`
3. **Typical off-diagonal value**: Should be `alpha*sim + P` ~ `0.05*0.5 + 1000` ≈ `1000`
4. **Energy magnitude**: For k=5 selections:
   - Diagonal: `5 * (-9000 + relevance)` ≈ `-45000`
   - Off-diagonal: `10 * (1000 + diversity)` ≈ `10000`
   - Total: `~-35000` + small adjustments for actual similarity

The penalty should still dominate to enforce cardinality, but similarity should matter at the margins.

## Additional Issue: Penalty Magnitude

Even with the fix, `P=1000` might be too large relative to similarity scores in `[0,1]`.

Consider:
- Similarity range: `[-1, 1]` (technically, but usually `[0, 1]` for cosine)
- Diagonal from similarity: `~[-1, 0]`
- Diagonal from penalty: `1000 * (-9) = -9000`

Ratio: `9000 : 1` is very large. Might need to reduce penalty or scale similarities.

However, this is less critical than the 4x bug. Let's fix the 4x issue first.
