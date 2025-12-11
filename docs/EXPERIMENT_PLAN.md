# Quantum RAG Experiment Plan

## Overview
A streamlined experimental validation of QUBO-based retrieval for RAG systems, demonstrating superiority over traditional methods (Top-K, MMR) in handling redundant information.

## Experimental Stages

### Stage 0: QUBO Mathematical Validation
**Goal:** Prove QUBO energy mathematically correlates with retrieval quality

**Experiment 0: Redundancy Robustness** (`exp_0_redundancy_robustness.py`)
- **Hypothesis:** QUBO energy consistently predicts distinct fact retrieval across all redundancy levels
- **Method:**
  - Enumerate all possible k-subsets for multiple prompts
  - Compute QUBO energy and distinct aspect count for each subset
  - Calculate Pearson correlation across redundancy levels 0-5
- **Success Criteria:**
  - Strong negative correlation (r < -0.7) at all redundancy levels
  - Proves QUBO energy is a valid objective function

---

### Stage 1: Baseline Method Comparison
**Goal:** Demonstrate QUBO maintains performance under redundancy while baselines degrade

**Experiment 1: Poisoned Stress Test** (`exp_1_poisoned_stress_test.py`)
- **Hypothesis:** Top-K degrades catastrophically with redundancy, MMR degrades moderately, QUBO remains robust
- **Method:**
  - Compare Top-K, MMR, and QUBO across redundancy levels 0-5
  - Measure aspect recall (% of distinct gold facts retrieved)
  - Test on 100 Wikipedia prompts with controlled redundancy injection
- **Success Criteria:**
  - QUBO: >85% aspect recall across all levels (robust)
  - MMR: 50-70% aspect recall at high redundancy (moderate degradation)
  - Top-K: <40% aspect recall at high redundancy (catastrophic degradation)
- **Notes:** This single experiment covers:
  - Clean data performance (level 0)
  - Stress testing (levels 1-5)
  - Dose-response relationship (continuous scaling)

---

### Stage 2: Parameter Sensitivity Analysis
**Goal:** Identify optimal QUBO parameters and validate robustness to parameter choices

**Experiment 2: Parameter Grid Search** (`exp_2_parameter_tuning.py`)
- **Hypothesis:** QUBO performance is robust across reasonable parameter ranges
- **Method:**
  - Grid search over α (diversity weight) and P (penalty) parameters
  - Test α ∈ [0.0, 0.1] and P ∈ {0.1, 1, 10, 100}
  - Measure average aspect recall across redundancy levels
- **Success Criteria:**
  - Identify optimal parameter region (α ≈ 0.05-0.08, P ≈ 1-10)
  - Show >10% parameter range maintains >80% of optimal performance
- **Output:** Heatmap showing parameter space and recommended values

---

### Stage 3: Scalability Analysis
**Goal:** Demonstrate QUBO efficiency scales to real-world dataset sizes

**Experiment 3: Scaling Benchmark** (`exp_3_scaling_benchmark.py`)
- **Hypothesis:** QUBO solver runtime remains practical as dataset size increases
- **Method:**
  - Test all methods (Top-K, MMR, QUBO) on increasing dataset sizes
  - Measure: retrieval quality, runtime, memory usage
  - Scale: 50, 100, 500, 1000, 5000 candidate chunks
  - Test both Gurobi and ORBIT QUBO solvers
- **Success Criteria:**
  - QUBO maintains quality advantage at all scales
  - Runtime remains <5s per query at 5000 candidates
  - Memory footprint stays reasonable (<4GB)
- **Output:** Scaling curves for quality, time, and memory

---

### Stage 4: Real-World Application Validation
**Goal:** Validate QUBO on actual RAG use cases beyond synthetic benchmarks

**Experiment 4: Production Simulation** (`exp_4_production_test.py`)
- **Hypothesis:** QUBO provides measurable improvement in real RAG applications
- **Method:**
  - Test on diverse domains: medical, legal, technical documentation
  - Compare end-to-end RAG quality (retrieval + generation)
  - Measure: factual accuracy, response completeness, hallucination rate
  - Use realistic redundancy patterns from actual knowledge bases
- **Success Criteria:**
  - QUBO improves answer quality by ≥15% over Top-K baseline
  - Reduces hallucinations by ≥20%
  - Increases fact coverage by ≥25%
- **Output:** Comprehensive comparison on real-world queries

---

## Experiment Dependencies

```
Stage 0 (Validation)
    └── Validates QUBO formulation
        └── Stage 1 (Comparison)
            └── Proves practical advantage
                ├── Stage 2 (Parameters)
                │   └── Optimizes configuration
                ├── Stage 3 (Scaling)
                │   └── Validates efficiency
                └── Stage 4 (Application)
                    └── Validates real-world impact
```

## Current Implementation Status

| Stage | Experiment | File | Status |
|-------|-----------|------|--------|
| 0 | Redundancy Robustness | `exp_0_redundancy_robustness.py` | ✅ Complete |
| 1 | Poisoned Stress Test | `exp_1_poisoned_stress_test.py` | ✅ Complete |
| 2 | Parameter Tuning | `exp_2_parameter_tuning.py` | ✅ Complete |
| 3 | Scaling Benchmark | `exp_3_scaling_benchmark.py` | 🔨 To Build |
| 4 | Production Test | `exp_4_production_test.py` | 🔨 To Build |

## Removed Experiments (Consolidated)

- ~~`exp_0_1_qubo_validation.py`~~ → Covered by exp_0
- ~~`exp_1_2_clean_control_test.py`~~ → Covered by exp_1 (level 0)
- ~~`exp_1_3_dose_response_curve.py`~~ → Covered by exp_1 (all levels)

## Key Metrics

1. **Aspect Recall**: % of distinct gold facts retrieved (primary metric)
2. **QUBO Energy**: Mathematical objective value (validation metric)
3. **Runtime**: Query processing time (efficiency metric)
4. **Memory**: Peak memory usage (scalability metric)
5. **End-to-End Quality**: RAG answer correctness (application metric)

## Dataset Structure

- **Wikipedia Dataset**: 100 prompts, 5 gold aspects each
- **Redundancy Levels**: 0 (clean), 1-5 (increasing redundancy)
- **Noise Chunks**: 25 per prompt (constant distractor set)
- **Total Chunks**: ~6000 (varying by redundancy level)

## Quick Reference

**Generate dataset (if needed):**
```bash
cd data/wikipedia
python generate_dataset.py --num-articles 100 --max-redundancy 5
```

**Run all core experiments:**
```bash
python experiments/exp_0_redundancy_robustness.py --num-prompts 10
python experiments/exp_1_poisoned_stress_test.py --num-prompts 100
python experiments/exp_2_parameter_tuning.py --num-prompts 10
```

**Expected runtime:**
- Dataset generation: ~60 min (100 articles)
- Stage 0: ~5 min (10 prompts)
- Stage 1: ~15 min (100 prompts)
- Stage 2: ~30 min (grid search)
