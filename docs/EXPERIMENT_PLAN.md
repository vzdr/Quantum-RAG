# Quantext Comprehensive Experiment Plan

**Objective:** Systematically prove that PC-QUBO outperforms Standard RAG across four dimensions: Quality, Efficiency, Scalability, and Business Impact.

**Competition Thesis:** Probabilistic Computing is the only scalable architecture for diversity-aware RAG in redundant data environments.

---

## Overview

This plan contains **9 experiments** organized into 5 parts:
- **Part 0:** QUBO mathematical validation (1 experiment)
- **Part 1:** Retrieval effectiveness (3 experiments)
- **Part 2:** Token efficiency (2 experiments)
- **Part 3:** Scalability & latency (2 experiments)
- **Part 4:** End-to-end business value (1 experiment)

**Timeline:** 14 days (2 weeks)

---

## Part 0: Foundation - QUBO Validation

### Experiment 0.1: Energy-Quality Correlation

**Purpose:** Validate that QUBO energy mathematically correlates with retrieval quality.

**Hypothesis:** Lower QUBO energy directly predicts higher percentage of distinct gold facts retrieved. The formulation has theoretical grounding, not arbitrary filtering.

**Method:**
1. For small candidate pools (N=30), enumerate all possible K=5 subsets via brute force
2. For each subset, compute:
   - QUBO energy E = -α·Σ(relevance) + β·Σ(redundancy)
   - Gold fact percentage (percentage of distinct gold facts retrieved)
3. Plot: Gold Fact % (x-axis) vs Normalized QUBO Energy (y-axis) with error bars

**Dataset:** 100 prompts from Wikipedia dataset, redundancy level 0 only (5 gold base + 25 noise)

**Settings:**
- N = 30 (candidate pool size: 5 gold + 25 noise)
- K = 5 (context window)
- Total subsets per prompt: C(30,5) = 142,506
- Energies normalized to [0,1] per prompt

**Implementation:** `experiments/exp_0_1_qubo_validation.py`

**Success Metric:**
- Pearson correlation r < -0.7 (strong negative correlation)
- Lower energy consistently predicts higher gold fact retrieval

**Output:** Plot with mean ± std dev error bars and line of best fit

**Narrative Value:** "QUBO energy is not arbitrary—it's a direct mathematical measure of retrieval quality. This validates our formulation before empirical testing."

**Timeline:** Days 1-2

---

### Experiment 0.2: Redundancy Robustness

**Purpose:** Validate QUBO energy correlation holds across all redundancy levels 0-5.

**Hypothesis:** QUBO energy consistently predicts distinct gold fact retrieval regardless of redundancy level, proving parameters aren't overfit to the no-redundancy case.

**Method:**
1. For each redundancy level (0-5):
   - Sample random prompts
   - Filter chunks to include gold (base + redundant up to level N) + noise
   - Enumerate combinations, compute energies
   - Normalize energies per prompt per level
   - Plot distinct facts vs energy for that level

**Dataset:** Random subset of Wikipedia prompts (10-100)

**Settings:**
- Redundancy levels: 0-5
- K = 5 chunks selected
- Candidate pool varies by level (e.g., level 0: 30 chunks, level 5: ~55 chunks)

**Implementation:** `experiments/exp_0_2_redundancy_robustness.py`

**Success Metric:**
- Correlation r < -0.7 for all redundancy levels
- Proves QUBO is robust to varying redundancy

**Output:** 6 separate plots (one per redundancy level) showing energy vs distinct facts

**Narrative Value:** "QUBO works consistently across all redundancy scenarios—not just the easy cases."

**Timeline:** Days 1-2

---

## Part 1: Effectiveness - Retrieval Quality

### Experiment 1.1: The Poisoned Stress Test

**Purpose:** Prove Top-K fails catastrophically under redundancy, while QUBO rescues distinct facts.

**Hypothesis:** Top-K retrieves duplicates of the same aspect and misses other aspects. QUBO rejects duplicates and retrieves distinct aspects.

**Dataset:** Wikipedia dataset with varying redundancy levels
- 100 prompts with 5 distinct aspects each
- Redundancy levels 0-5 (0-5 redundant chunks per aspect)
- 25 noise chunks per prompt
- Total candidate pool: 30-55 paragraphs per prompt

**Settings:**
- Retrieval pool: N = 100
- Context window: K = 5
- QUBO parameters: α = 0.7, β = 0.05

**Comparison:** Top-K vs QUBO diversity retrieval

**Implementation:** `scripts/run_wikipedia_experiment.py`

**Metrics:**
- **Aspect Recall:** Percentage of prompts where all 5 distinct aspects retrieved
- **Gold Recall:** Percentage of individual gold facts retrieved
- **Redundancy Rate:** Percentage of retrieved chunks from same aspect
- **Redundancy Score:** Average pairwise cosine similarity in top-K

**Success Metrics:**
- QUBO Aspect Recall: >90%
- Top-K Aspect Recall: <30%
- QUBO Redundancy: <0.1
- Top-K Redundancy: >0.8

**Output:** Bar chart comparing Top-K vs QUBO on all metrics

**Narrative Value:** "Standard RAG breaks under redundancy. Our Wikipedia dataset simulates real-world enterprise data—and Top-K fails spectacularly."

**Timeline:** Days 3-4

---

### Experiment 1.2: The Clean Control Test

**Purpose:** Prove QUBO is safe and doesn't harm performance on clean data.

**Hypothesis:** On datasets without redundancy (level 0), QUBO performs comparably to Top-K, proving it doesn't hallucinate or delete valid information.

**Dataset:** Wikipedia prompts at redundancy level 0 only
- 100 prompts
- 5 gold base chunks + 25 noise per prompt
- No redundant chunks

**Settings:**
- N = 100
- K = 5
- Same QUBO parameters: α = 0.7, β = 0.05

**Comparison:** Top-K vs QUBO on clean data

**Implementation:** `scripts/run_clean_wikipedia_experiment.py`

**Metrics:**
- Aspect Recall
- Gold Recall
- Redundancy Score

**Success Metric:**
- QUBO recall within ±5% of Top-K recall
- Proves QUBO doesn't introduce false negatives

**Output:** Bar chart showing comparable performance

**Narrative Value:** "Our solution is safe and robust. QUBO only filters redundancy—it doesn't harm retrieval on clean data."

**Timeline:** Days 3-4

---

### Experiment 1.3: The Dose-Response Curve

**Purpose:** Show QUBO advantage scales continuously with redundancy level.

**Hypothesis:** Even mild redundancy degrades Top-K performance, while QUBO remains stable across all redundancy levels.

**Dataset:** Test across redundancy levels 0-5
- Same 100 prompts tested at each level
- Level 0: 5 gold base (no redundancy)
- Level 1: 5 gold base + 5 redundant (1 per aspect)
- Level 2: 5 gold base + 10 redundant (2 per aspect)
- Level 3: 5 gold base + 15 redundant (3 per aspect)
- Level 5: 5 gold base + 25 redundant (5 per aspect)

**Settings:**
- N = 100
- K = 5
- Test both Top-K and QUBO on each level

**Implementation:** `scripts/run_dose_response_experiment.py`

**Metrics:** Aspect Recall for each redundancy level

**Success Metrics:**
- Top-K recall drops >20% with just redundancy level 1
- QUBO recall remains flat within ±5% across all levels

**Output:** Line graph
- X-axis: Redundancy level (0, 1, 2, 3, 5)
- Y-axis: Aspect Recall (%)
- Two lines: Top-K (declining), QUBO (flat)

**Narrative Value:** "Even mild redundancy breaks Top-K. This isn't a worst-case scenario—it's inevitable in real-world data. QUBO is immune across all redundancy levels."

**Timeline:** Days 5-6

---

## Part 2: Efficiency - Token Economics

### Experiment 2.1: Information Density

**Purpose:** Quantify how much unique signal vs redundant noise fills the context window.

**Concept:** Calculate information density = Unique Gold Aspects / Total Tokens

**Method:**
1. For each retrieval method (Top-K, QUBO):
   - Count unique gold aspects in top-K results
   - Count total tokens in context window
   - Compute density ratio
2. Compare average density across 100 prompts

**Dataset:** Wikipedia dataset at redundancy level 5 (from Exp 1.1)

**Settings:** K = 5

**Implementation:** Add to `scripts/run_wikipedia_experiment.py`

**Success Metric:**
- QUBO density: >80% (minimal waste)
- Top-K density: <20% (high waste due to duplicates)

**Output:** Bar chart comparing information density

**Narrative Value:** "Top-K wastes 80% of the context window on duplicate tokens. QUBO is 4x more information-dense—same knowledge, fewer tokens."

**Timeline:** Days 8-9

---

### Experiment 2.2: The Cost of Recall Curve

**Purpose:** Show how much larger K needs to be for Top-K to match QUBO performance.

**Hypothesis:** To achieve the same recall as QUBO@K=5, Top-K requires K=20-25 (4-5x more tokens and cost).

**Method:**
1. Run QUBO at K=5, record Aspect Recall (e.g., 95%)
2. Run Top-K at K = 5, 10, 15, 20, 25, 30 until it matches 95% recall
3. Calculate token cost multiplier

**Dataset:** Wikipedia dataset at redundancy level 5

**Settings:** Variable K for Top-K, fixed K=5 for QUBO

**Implementation:** Loop in `scripts/run_wikipedia_experiment.py`

**Metrics:**
- Recall vs K curve
- Token count at equivalent recall
- Cost multiplier (Top-K tokens / QUBO tokens)

**Success Metric:** Top-K requires 4-5x more tokens to match QUBO recall

**Output:** Line graph
- X-axis: Context size K
- Y-axis: Aspect Recall %
- Horizontal line: QUBO baseline (K=5)
- Curve: Top-K performance increasing with K

**Narrative Value:** "To get the same accuracy, Standard RAG requires 5x more tokens—5x higher cost and latency. QUBO is the economically viable choice for production."

**Timeline:** Days 8-9

---

## Part 3: Scalability - The Latency Challenge

### Experiment 3.1: The Classical Latency Wall

**Purpose:** Prove classical solvers cannot handle QUBO optimization at scale required for real-time RAG.

**Hypothesis:** As candidate pool N grows, classical solvers slow down exponentially or become intractable.

**Solvers Tested:**
1. **Brute Force:** Exhaustive enumeration (exact solution)
   - Expected: Fails/timeout at N > 50
2. **Gurobi:** Commercial MILP solver (exact solution)
   - Expected: Timeout at N = 500-1000
3. **Simulated Annealing:** Heuristic approximation
   - Expected: Feasible but slow (>1s at N=1000)

**Settings:**
- Test N = [20, 50, 100, 200, 500, 1000]
- K = 5 (fixed)
- 50 prompts per N value
- Measure average latency per query

**Implementation:** `scripts/run_scalability_experiment.py`

**Metrics:**
- Latency (ms) vs N for each solver
- Solution quality (aspect recall) vs N

**Success Metrics:**
- Brute force: Timeout at N=50
- Gurobi: >5s latency at N=500
- Simulated Annealing: >1s at N=1000 (too slow for real-time)

**Output:** Line graph
- X-axis: Candidate pool size N
- Y-axis: Latency (ms, log scale)
- Three lines: Brute Force, Gurobi, SA

**Narrative Value:** "Classical computing hits a wall. Even the best commercial solvers (Gurobi) and heuristics (SA) are too slow for real-time RAG at enterprise scale."

**Timeline:** Days 10-11

---

### Experiment 3.2: The PC Advantage

**Purpose:** Demonstrate Probabilistic Computing maintains constant/linear latency at scale.

**Hypothesis:** PC hardware solves QUBO in <100ms even at N=1000, enabling real-time diversity-aware retrieval.

**Hardware:** Quantum Dice PC API (provided during competition)

**Settings:**
- Test same N range: [20, 50, 100, 200, 500, 1000]
- K = 5
- Same 50 prompts per N value
- ORBIT library integration

**Implementation:** Add PC solver to `scripts/run_scalability_experiment.py`

**Metrics:**
- PC latency vs N
- Solution quality comparison with classical solvers

**Success Metrics:**
- PC latency: <100ms at N=1000
- Maintains >90% aspect recall at all N values
- Latency growth: constant or linear (not exponential)

**Output:** Line graph overlay on Exp 3.1
- Same axes as 3.1
- Add fourth line: PC (flat/low latency)
- Highlight the gap at N=1000

**Narrative Value:** "Probabilistic Computing is the only architecture that enables real-time diversity-aware retrieval at scale. While classical solvers timeout, PC delivers <100ms responses at 1000 candidates."

**Timeline:** Days 10-11

---

## Part 4: Business Impact - End-to-End Validation

### Experiment 4.1: LLM Generation Quality

**Purpose:** Prove better retrieval translates to better answers—demonstrating real business value.

**Hypothesis:** QUBO's superior retrieval (distinct aspects) enables accurate LLM answers, while Top-K's redundant retrieval causes hallucinations or refusals.

**Method:**
1. For 100 Wikipedia prompts:
   - **Pipeline A:** Top-K retrieval → GPT-4o-mini generation → Answer
   - **Pipeline B:** QUBO retrieval → GPT-4o-mini generation → Answer
2. Compare generated answers against expected answers (based on retrieved aspects)

**Dataset:** Wikipedia dataset at redundancy level 5 (100 prompts)

**Settings:**
- Retrieval: N=100, K=5
- LLM: GPT-4o-mini
- Same prompt template for both pipelines
- Temperature: 0 (deterministic)

**Implementation:** `scripts/run_end_to_end_experiment.py`

**Metrics:**
1. **Aspect Coverage:** How many of the 5 aspects are reflected in the answer?
2. **Answer Completeness:** F1 score based on expected comprehensive answer
3. **"Unable to Answer" Rate:** How often does LLM refuse/say "insufficient information"?
4. **Hallucination Rate:** Factual errors or information not in retrieved chunks

**Success Metrics:**
- QUBO Aspect Coverage: >90%
- Top-K Aspect Coverage: <40%
- QUBO Completeness F1: >85%
- Top-K "Unable to Answer": >30%

**Output:** Bar charts
1. Aspect Coverage comparison
2. Answer quality breakdown (Complete / Incomplete / Unable / Hallucination)

**Narrative Value:** "Better retrieval doubles answer quality. QUBO's diversity-aware approach directly translates to better business outcomes—fewer hallucinations, more confident answers, higher user satisfaction."

**Timeline:** Days 12-13

---

## Summary of Deliverables

| # | Experiment | Script | Output | Key Metric | Timeline |
|---|------------|--------|--------|-----------|----------|
| **0.1** | QUBO Validation | exp_0_1_qubo_validation.py | Energy-Quality Plot | r < -0.7 | Days 1-2 |
| **0.2** | Redundancy Robustness | exp_0_2_redundancy_robustness.py | 6 plots (one per level) | r < -0.7 all levels | Days 1-2 |
| **1.1** | Poisoned Test | run_wikipedia_experiment.py | Recall Bar Chart | >90% vs <30% | Days 3-4 |
| **1.2** | Clean Control | run_clean_wikipedia_experiment.py | Recall Bar Chart | Within ±5% | Days 3-4 |
| **1.3** | Dose-Response | run_dose_response_experiment.py | Recall vs Level Line | Flat vs Declining | Days 5-6 |
| **2.1** | Information Density | (Modify Exp 1.1) | Density Bar Chart | 80% vs 20% | Days 8-9 |
| **2.2** | Cost of Recall | (Loop Exp 1.1 with K) | Recall vs K Curve | 5x token savings | Days 8-9 |
| **3.1** | Classical Latency | run_scalability_experiment.py | Latency Line (3 solvers) | >1s at N=1000 | Days 10-11 |
| **3.2** | PC Advantage | (Add PC to 3.1) | Latency Overlay | <100ms at N=1000 | Days 10-11 |
| **4.1** | LLM Quality | run_end_to_end_experiment.py | Aspect Coverage Chart | 90% vs 40% | Days 12-13 |

---

## Implementation Timeline (14 Days)

### Week 1: Foundation & Core Quality

**Days 1-2:** Foundation
- ✓ Set up data pipeline (complete)
- ✓ Implement QUBO validation (Exp 0.1)
- ✓ Implement redundancy robustness (Exp 0.2)
- ✓ Verify mathematical correlation

**Days 3-4:** Retrieval Effectiveness
- Implement poisoned test (Exp 1.1)
- Implement clean control (Exp 1.2)
- Generate baseline visualizations

**Days 5-6:** Dose-Response Analysis
- Test across all redundancy levels
- Run dose-response experiment (Exp 1.3)
- Validate continuous relationship

**Day 7:** Week 1 Checkpoint
- Review all quality experiments
- Validate visualizations
- Prepare Week 2 infrastructure

### Week 2: Efficiency, Scalability & Impact

**Days 8-9:** Token Efficiency
- Add density calculations (Exp 2.1)
- Run cost-of-recall analysis (Exp 2.2)
- Generate economic impact metrics

**Days 10-11:** Scalability Testing
- Implement classical solvers (Exp 3.1)
- Integrate ORBIT/PC API (Exp 3.2)
- Generate latency comparisons

**Days 12-13:** End-to-End Validation
- Implement LLM pipeline (Exp 4.1)
- Run GPT-4o-mini experiments
- Calculate answer quality metrics

**Day 14:** Final Analysis & Submission
- Compile all visualizations
- Write final narratives
- Prepare competition submission

---

## Scientific Narrative Arc

### Act 1: Foundation (Exp 0.1-0.2)
"Our QUBO formulation has mathematical rigor—energy correlates with quality across all redundancy scenarios."

### Act 2: The Problem (Exp 1.1-1.3)
"Standard RAG breaks under redundancy. Even mild duplication causes catastrophic failure. This is inevitable in real-world enterprise data."

### Act 3: The Solution Depth (Exp 2.1-2.2)
"QUBO doesn't just fix retrieval—it's 4x more token-efficient. Same quality, 80% less cost."

### Act 4: The Scalability Barrier (Exp 3.1)
"But classical computing can't solve QUBO at scale. Every solver—exact or heuristic—hits a wall."

### Act 5: The Enabler (Exp 3.2)
"Probabilistic Computing breaks through. Where classical times out, PC delivers <100ms at 1000 candidates."

### Act 6: The Impact (Exp 4.1)
"Better retrieval = better answers. QUBO doubles LLM answer quality. This is real business value."

---

## Key Design Principles

1. **Mathematical Rigor:** Start with validation (Exp 0.1-0.2) before empirical tests
2. **Continuous Relationships:** Use dose-response (Exp 1.3) to show scaling, not cherry-picked extremes
3. **Safety Controls:** Include clean data test (Exp 1.2) to prove no harm
4. **Economic Relevance:** Measure token efficiency (Part 2), not just accuracy
5. **Exhaustive Comparison:** Test multiple classical solvers (Part 3), not just one
6. **Business Validation:** End-to-end LLM quality (Exp 4.1) proves real-world value

---

## Success Criteria

**Minimum Viable Results (to compete):**
- Exp 1.1: QUBO >70% recall, Top-K <40%
- Exp 3.2: PC latency <200ms at N=1000

**Target Results (to win):**
- Exp 0.1-0.2: Strong negative correlation (r < -0.7) across all redundancy levels
- Exp 1.1: QUBO >90% recall, Top-K <30%
- Exp 2.2: 5x token savings demonstrated
- Exp 3.2: PC <100ms at N=1000
- Exp 4.1: QUBO Aspect Coverage >90%, Top-K <40%

**Stretch Goals (publication-quality):**
- All 9 experiments complete with visualizations
- Dose-response curve showing continuous scaling
- Multi-solver classical comparison
- Statistical significance testing (p<0.01)

---

## Dependencies & Prerequisites

**Data:**
- ✓ Wikipedia articles (632 curated)
- ✓ Chroma database (5,600 chunks)
- ✓ BGE embeddings (generated)
- ✓ Similarity matrix precomputed
- ✓ Redundancy levels 0-5 (generated)

**Models:**
- ✓ BGE-large-en-v1.5 (embeddings)
- ⚠ GPT-4o-mini (need API key for Exp 4.1)

**Solvers:**
- ⚠ Brute force (implement)
- ⚠ Gurobi (requires license or trial)
- ⚠ Simulated Annealing (implement)
- ⚠ ORBIT/PC API (competition access needed)

**Infrastructure:**
- ✓ Chroma database (local persistent storage)
- ✓ GPU available (RTX 5060, 8GB)
- ✓ Precomputed similarity matrices

---

## Next Steps

1. **Immediate (Today):**
   - Review this comprehensive plan
   - Confirm timeline feasibility
   - Identify any blockers (API access, licenses)

2. **Days 1-2:** ✓ Complete
   - ✓ Implement Experiment 0.1 (QUBO validation)
   - ✓ Implement Experiment 0.2 (Redundancy robustness)
   - ✓ Set up experiment infrastructure
   - ✓ Create visualization templates

3. **Week 1 Goal:**
   - Complete all quality experiments (0.1, 0.2, 1.1, 1.2, 1.3)
   - Generate first set of visualizations
   - Validate core thesis

4. **Week 2 Goal:**
   - Complete efficiency and scalability experiments
   - Integrate PC hardware
   - Run end-to-end validation
   - Prepare final submission

---

This comprehensive plan provides a complete scientific argument: **Quality → Efficiency → Scalability → Impact**. Every experiment builds toward the central thesis: *Probabilistic Computing is the only viable architecture for diversity-aware RAG at enterprise scale.*
