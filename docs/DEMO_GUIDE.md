# QUBO-RAG Medical Diagnosis Demo Guide

## Overview
This demo showcases how QUBO-RAG's diversity optimization provides superior retrieval for medical diagnosis queries compared to Naive (Top-K) and MMR approaches.

## What Was Changed

### 1. Medical Diagnosis Dataset
- **Location:** `data/medical/raw/`
- **Content:** 60 medical document chunks covering 6 diseases
- **Diseases:**
  - Lupus (SLE)
  - Rheumatoid Arthritis
  - Lyme Disease
  - Fibromyalgia
  - Chronic Fatigue Syndrome (ME/CFS)
  - Hypothyroidism

**Key Design:** Strategic symptom overlap (fatigue, joint pain, fever) creates realistic retrieval challenges where Naive RAG gets "stuck" on the most common symptom.

### 2. Updated Notebook (RAG_System.ipynb)
- **Cell 5:** Loads medical diagnosis dataset instead of LOTR
- **Cell 13:** Uses medical diagnosis query instead of "What is machine learning?"
- **Cell 19-25:** Updated comparison section with medical query
- **New Cells (after Cell 27):**
  - Disease Coverage Visualization (bar chart)
  - LLM Response Quality Comparison (side-by-side)
  - Summary Table (quantitative metrics)

### 3. Enhanced Document Loader
- Automatically extracts disease name from filename (e.g., `lupus_symptoms_1.txt` → `disease: "lupus"`)
- Enables disease-level analysis in retrieval results

---

## How to Run the Demo

### Step 1: Ensure Medical Data Exists
```bash
# Check that medical documents were generated
ls data/medical/raw/*.txt | wc -l  # Should show 60
```

If not, run:
```bash
cd data/medical/raw
python generate_medical_data.py
```

### Step 2: Run the Notebook
Open `RAG_System.ipynb` in Jupyter and run cells in order:

1. **Cells 1-3:** Setup and imports
2. **Cells 4-9:** Index medical documents (Phase A)
   - Should show 60 medical documents loaded
   - Creates ~60 chunks (may vary with chunking strategy)
   - Generates embeddings and stores in vector database

3. **Cells 10-14:** Single query test (Phase B)
   - Tests basic retrieval on medical query
   - Verifies system is working

4. **Cells 18-27:** Method comparison (the main demo!)
   - Compares Naive vs MMR vs QUBO-RAG
   - Shows diversity metrics

5. **New Cells (after 27):**
   - **Disease Coverage Chart:** Visual proof of QUBO's diversity
   - **Response Comparison:** Shows LLM output quality difference
   - **Summary Table:** Quantitative metrics

---

## Expected Results

### Quantitative Metrics

| Method | Expected Diseases Covered | Expected Intra-List Similarity | Expected Behavior |
|--------|---------------------------|-------------------------------|-------------------|
| **Naive (Top-K)** | 1-2 diseases | High (>0.65) | Focuses on most common symptom (e.g., all fatigue docs) |
| **MMR** | 2-3 diseases | Medium (~0.45-0.55) | Balances relevance and diversity |
| **QUBO-RAG** | 4-5 diseases | Low (<0.40) | Optimal diversity across disease clusters |

### Qualitative Differences

**Naive Response:**
- Narrow focus on 1-2 diseases
- Example: "This appears to be Chronic Fatigue Syndrome based on..."
- Misses important differential diagnoses

**MMR Response:**
- Mentions 2-3 possible conditions
- Better than Naive but still limited

**QUBO-RAG Response:**
- Comprehensive differential diagnosis covering 4-5 diseases
- Example: "Several conditions should be considered: Lupus (SLE) presents with..., Rheumatoid Arthritis shows..., Lyme disease may cause..., Chronic Fatigue Syndrome..., Hypothyroidism can manifest..."
- Provides distinguishing features for each
- Clinically most useful for differential diagnosis

---

## Test Queries

The primary demo query is:
```
"Patient presents with chronic fatigue, joint pain, and occasional low-grade fever.
What conditions should be considered in the differential diagnosis?"
```

**Why this query works well:**
- All three symptoms (fatigue, joint pain, fever) appear across multiple diseases
- Naive retrieval gets stuck on "fatigue" (most common symptom)
- QUBO retrieves diverse disease perspectives

### Additional Test Queries

You can also try:

**Query 2: Rash + Systemic**
```python
test_query = "Patient has a rash, fever, and severe fatigue. What could be causing these symptoms?"
```
Expected: Naive focuses on one rash type; QUBO covers Lupus + Lyme + systemic causes

**Query 3: Diagnostic Confusion**
```python
test_query = "Patient reports morning stiffness, widespread pain, and chronic tiredness. How can we distinguish between possible diagnoses?"
```
Expected: Naive focuses on RA; QUBO covers RA + Fibromyalgia + Lupus with diagnostic criteria

---

## Troubleshooting

### Issue: "All methods show similar intra-list similarity"

**Cause:** Synthetic data may not have enough overlap, or k is too small.

**Solutions:**
1. Increase k (try k=10 instead of k=5)
2. Tune QUBO alpha parameter (try 0.4-0.7 range in Cell 25)
3. Retrieve more candidates (change `k*3` to `k*5` in Cell 21)

### Issue: "QUBO doesn't cover more diseases than MMR"

**Cause:** QUBO solver parameters may need tuning.

**Solutions:**
1. Increase ORBIT sweeps: `'full_sweeps': 10000` (currently 5000)
2. Adjust alpha parameter: Try 0.5 or 0.7 instead of 0.6
3. Use more replicas: `'n_replicas': 4` (currently 2)

### Issue: "LLM responses are too similar despite diverse retrieval"

**Cause:** LLM temperature causing variation.

**Solutions:**
1. Set temperature=0 in Cell 3 config for deterministic responses
2. Update prompt to explicitly request differential diagnosis
3. Show retrieved chunks as evidence alongside LLM response

### Issue: "No disease metadata in results"

**Cause:** Document loader not extracting disease names.

**Solutions:**
1. Verify disease metadata extraction in `core/document_loader.py` (lines 103-116)
2. Check that filenames follow pattern: `{disease}_{category}_{num}.txt`
3. Re-run Cell 5 to reload documents with updated loader

---

## Performance Tuning

### For Faster Demos (Development)
```python
# In Cell 25, reduce ORBIT sweeps:
solver_params={'n_replicas': 2, 'full_sweeps': 1000}  # ~1-2s instead of ~5-10s
```

### For Best Quality (Presentation)
```python
# In Cell 25, increase ORBIT sweeps:
solver_params={'n_replicas': 4, 'full_sweeps': 10000}  # ~10-20s but better quality
```

### For Most Dramatic Difference
1. Use k=10 (more chunks to show diversity)
2. Set alpha=0.5 (stronger diversity emphasis)
3. Retrieve from larger candidate pool: `k*5` instead of `k*3`

---

## Presentation Tips

1. **Start with the problem:**
   - "Medical diagnosis often involves overlapping symptoms"
   - "Naive RAG retrieves redundant information about the most common symptom"
   - "We need diverse context for comprehensive differential diagnosis"

2. **Show the disease coverage chart:**
   - Visual proof that QUBO retrieves from more diseases
   - Clear bar chart comparison

3. **Compare LLM responses:**
   - Read Naive response (narrow focus)
   - Read MMR response (better but still limited)
   - Read QUBO response (comprehensive differential)
   - Highlight how diverse retrieval leads to better clinical advice

4. **Emphasize metrics:**
   - Show summary table
   - Point out lower intra-list similarity for QUBO
   - Connect metrics to clinical value

5. **The punchline:**
   - "QUBO-RAG balances relevance AND diversity optimally"
   - "For medical queries, this means better differential diagnosis"
   - "Diverse context → Comprehensive LLM response → Better clinical decision support"

---

## Next Steps for Further Improvement

1. **Real Benchmark Validation:**
   - Test on BEIR medical datasets (TREC-COVID, NFCorpus)
   - Compare against published MMR baselines
   - Measure alpha-nDCG on standard benchmarks

2. **More Realistic Synthetic Data:**
   - Increase to 100-200 documents
   - Add more diseases (8-10 total)
   - Include atypical presentations

3. **Timing Benchmarks:**
   - Measure actual solve time (should be <50ms with optimized params)
   - Compare against MMR timing
   - Profile ORBIT performance

4. **User Study:**
   - Have medical professionals rate response quality
   - Blind comparison of Naive vs QUBO responses
   - Measure diagnostic accuracy/completeness

5. **Production Readiness:**
   - Build REST API
   - Add caching for common queries
   - Implement batch processing
   - Add monitoring/logging

---

## Success Criteria

The demo successfully demonstrates QUBO-RAG advantage if:

✅ **Disease Coverage:** QUBO retrieves from ≥4 diseases vs Naive's 1-2
✅ **Diversity:** QUBO intra-list similarity <0.40 vs Naive's >0.65
✅ **Response Quality:** QUBO response mentions 4+ diseases in differential
✅ **Visual Impact:** Bar chart clearly shows QUBO's broader coverage
✅ **Narrative:** Progressive reveal (Naive → MMR → QUBO) tells compelling story

If any criteria aren't met, refer to Troubleshooting section above.

---

## File Reference

- **Medical Data:** `data/medical/raw/*.txt` (60 files)
- **Data Generator:** `data/medical/raw/generate_medical_data.py`
- **Notebook:** `RAG_System.ipynb`
- **Document Loader:** `core/document_loader.py` (disease metadata extraction)
- **QUBO Solver:** `core/qubo_solver.py`
- **Retrieval Strategies:** `core/retrieval_strategies.py`
- **Diversity Metrics:** `core/diversity_metrics.py`

---

## Contact / Questions

If the demo doesn't show clear QUBO advantage:
1. Check Expected Results section
2. Try Troubleshooting solutions
3. Tune parameters (alpha, sweeps, k)
4. Consider regenerating synthetic data with more overlap

For best results: Use k=10, alpha=0.5-0.6, full_sweeps=5000-10000
