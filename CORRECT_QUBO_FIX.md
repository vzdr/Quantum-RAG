# Correct QUBO Cardinality Constraint Fix

## The Correct Formulation

For cardinality constraint `Σxᵢ = k`, we penalize with:

```
P * (Σᵢ xᵢ - k)²
```

Expanding:
```
= P * [(Σᵢ xᵢ)² - 2k(Σᵢ xᵢ) + k²]
= P * [Σᵢ xᵢ² + Σᵢ≠ⱼ xᵢxⱼ - 2k Σᵢ xᵢ + k²]
= P * [Σᵢ xᵢ + Σᵢ≠ⱼ xᵢxⱼ - 2k Σᵢ xᵢ + k²]    (binary: xᵢ² = xᵢ)
= P * [Σᵢ (1-2k)xᵢ + Σᵢ≠ⱼ xᵢxⱼ] + Pk²
```

Ignoring constant `Pk²`:
- **Linear coefficient**: `P(1 - 2k)` for each `xᵢ`
- **Quadratic coefficient**: `P` for each pair `xᵢxⱼ` where `i ≠ j`

## QUBO Matrix Representation

The QUBO is: `minimize E = x^T Q x`

For a **symmetric** matrix Q:
```
x^T Q x = Σᵢ Q[i,i]xᵢ² + Σᵢ<ⱼ Q[i,j]xᵢxⱼ + Σⱼ<ᵢ Q[i,j]xᵢxⱼ
        = Σᵢ Q[i,i]xᵢ + 2 Σᵢ<ⱼ Q[i,j]xᵢxⱼ    (using xᵢ² = xᵢ and symmetry)
```

So to get coefficient `C` for a pair `xᵢxⱼ` where `i < j`:
```
2 * Q[i,j] = C
Q[i,j] = C/2
```

## Current Bug

From `core/qubo_solver.py`:

```python
# Line 84: Diagonal
Q[i, i] += self.penalty * (1 - 2 * self.k)  # ✓ CORRECT: P(1-2k)

# Line 93: Off-diagonal
Q[i, j] += 2 * self.penalty  # ❌ WRONG: Should be P/2, not 2P!
```

The off-diagonal term `2 * self.penalty` gives:
- Contribution to energy: `2 * (2P) = 4P` per pair
- Should be: `P` per pair
- Error factor: **4x too large**

## The Correct Fix

```python
# core/qubo_solver.py, lines 86-96

for i in range(self.n):
    for j in range(i + 1, self.n):
        # Diversity term (weighted by alpha)
        Q[i, j] = self.alpha * self.pairwise_sims[i, j]

        # Cardinality penalty: P/2 (because symmetric matrix doubles it)
        Q[i, j] += self.penalty / 2.0  # FIXED!

        # Ensure symmetry
        Q[j, i] = Q[i, j]
```

## Why This Matters

With k=5, P=1000:
- **Before fix**: Off-diagonal contribution = `10 pairs * 4*1000 = 40,000`
- **After fix**: Off-diagonal contribution = `10 pairs * 1000 = 10,000`

This 4x reduction means:
- The penalty still enforces cardinality (as intended)
- But doesn't completely overwhelm the similarity signal
- ORBIT solver can actually find meaningful solutions

## Verification

After the fix, for k=5 selections:

**Diagonal terms**:
- From similarity: `~[-1, 0]`
- From penalty: `P(1-2k) = 1000(-9) = -9000`
- Total: `~-9000` per selected item

**Off-diagonal terms** (for symmetric matrix):
- From diversity: `α * sim[i,j] ~ 0.05 * 0.5 = 0.025`
- From penalty: `P/2 = 500`
- Total per matrix element: `~500`
- Total contribution to energy: `2 * 500 = 1000` per pair (symmetric matrix doubles it)

**Total energy for k=5**:
- Diagonal: `5 * (-9000) = -45,000`
- Off-diagonal: `C(5,2) * 1000 = 10 * 1000 = 10,000`
- Net: `~-35,000`
- Plus similarity adjustments: `~[-5, 0]`

The penalty is still strong (to enforce k=5), but similarity matters!

## Alternative: Simpler Formulation

If this is still confusing, we could use a simpler approach:

```python
# Only fill upper triangle, don't symmetrize
for i in range(self.n):
    for j in range(i + 1, self.n):
        Q[i, j] = self.alpha * self.pairwise_sims[i, j] + self.penalty
        # Don't set Q[j, i]

# Compute energy manually as sum over upper triangle only
def energy(x, Q):
    E = np.dot(x, np.diagonal(Q))  # Linear terms
    E += 2 * np.sum(Q[np.triu_indices_from(Q, k=1)] *
                    np.outer(x, x)[np.triu_indices_from(Q, k=1)])
    return E
```

But this requires changing how ORBIT is called. The symmetric matrix approach is standard, we just need to account for the factor of 2.
