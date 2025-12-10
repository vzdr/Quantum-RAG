# Wikipedia Dataset Integration

This document describes the integration of the Wikipedia dataset from Quantum Dice v2 into the Quantum-RAG framework.

## Overview

The integrated repository combines:
- **Quantum-RAG**: Advanced QUBO-based diversity-aware retrieval with ORBIT solver
- **Quantum Dice v2 Dataset**: Wikipedia articles with controlled redundancy levels for testing

## Directory Structure

```
Quantum-RAG copy/
├── core/
│   ├── wikipedia/          # NEW: Wikipedia fetching utilities
│   │   ├── __init__.py
│   │   └── fetcher.py     # Wikipedia article fetcher with caching
│   ├── analysis_utils.py  # NEW: Evaluation metrics and analysis
│   ├── embedder.py        # UPDATED: Added BGE-large model support
│   └── vector_store.py    # UPDATED: Added metadata filtering
│
├── data/
│   ├── medical_diagnosis/ # Original medical dataset
│   └── wikipedia/         # NEW: Wikipedia dataset from QD v2
│       ├── wikipedia_db/     # Pre-built ChromaDB (5,600 chunks)
│       ├── cache/         # Cached Wikipedia articles
│       ├── checkpoints/   # Pipeline checkpoints
│       ├── similarity/    # Pre-computed similarity matrices
│       └── wiki_articles.txt  # List of 675 articles
│
└── config/
    └── settings.py        # UPDATED: Added dataset_type parameter
```

## What Was Integrated

### 1. Wikipedia Dataset (from Quantum Dice v2)
- **ChromaDB Database**: 5,600 chunks with controlled redundancy
  - 100 prompts (queries)
  - 500 gold base chunks (5 aspects × 100 prompts)
  - 2,500 gold redundant chunks
  - 2,500 noise chunks
- **Embeddings**: BAAI/bge-large-en-v1.5 (1024-dimensional)
- **Similarity Matrices**: Pre-computed 5600×5600 similarity matrix (239 MB)
- **Cache**: Cached Wikipedia articles for re-processing

### 2. New Modules

#### `core/wikipedia/fetcher.py`
Wikipedia article fetcher with:
- Automatic caching (MD5-based filenames)
- Retry logic with exponential backoff
- Article quality validation
- Section extraction and filtering

#### `core/analysis_utils.py`
Evaluation utilities:
- `compute_pairwise_similarities()`: Generate similarity matrices
- `evaluate_retrieval_quality()`: Compute gold recall, redundancy metrics
- `compute_qubo_energy()`: Calculate QUBO energy for selections

### 3. Enhanced Modules

#### `core/embedder.py`
Added support for `BAAI/bge-large-en-v1.5` model:
```python
generator = EmbeddingGenerator(model_name='BAAI/bge-large-en-v1.5')
```

#### `core/vector_store.py`
New methods for advanced filtering:
- `get_by_metadata(where)`: Get chunks by metadata criteria
- `search_with_filters(...)`: Advanced search with inclusion/exclusion filters

#### `config/settings.py`
Added `dataset_type` parameter:
```python
config = RAGConfig(
    dataset_type="wikipedia",  # or "medical"
    embedding_model="BAAI/bge-large-en-v1.5",
    collection_name="wiki_aspects",
    persist_directory="./data/wikipedia/wikipedia_db"
)
```

## Using the Wikipedia Dataset

### Option 1: Load Pre-built ChromaDB (Recommended)

The dataset is already embedded and stored in ChromaDB with BGE-large embeddings:

```python
from core.vector_store import VectorStore
from core.embedder import EmbeddingGenerator

# Initialize with Wikipedia dataset
vector_store = VectorStore(
    collection_name="wiki_aspects",
    persist_directory="./data/wikipedia/wikipedia_db"
)

# Use BGE-large model for queries (must match dataset embeddings)
embedder = EmbeddingGenerator(
    model_name="BAAI/bge-large-en-v1.5",
    device="cpu"  # or "cuda"
)

# Query the dataset
query = "Explain the history of the Roman Empire"
query_embedding = embedder.embed_query(query)

# Basic search
results = vector_store.search(query_embedding, k=10)

# Search with metadata filtering (e.g., redundancy level)
results = vector_store.search_with_filters(
    query_embedding,
    k=10,
    metadata_filter={"redundancy_index": {"$lte": "2"}},  # Base + redundancy 0-2
    exclude_metadata={"chunk_type": "noise"}  # Exclude noise
)
```

### Option 2: Use Analysis Utilities

```python
from core.analysis_utils import evaluate_retrieval_quality, compute_pairwise_similarities
import numpy as np

# Load precomputed similarity matrix
sim_data = np.load('./data/wikipedia/similarity/similarity_matrix.npz')
similarity_matrix = sim_data['similarity_matrix']

# Evaluate retrieval quality
selected_indices = [0, 5, 10, 15, 20]
gold_indices = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]  # Ground truth

metrics = evaluate_retrieval_quality(
    selected_indices=selected_indices,
    gold_indices=gold_indices,
    similarity_matrix=similarity_matrix
)

print(f"Gold Recall: {metrics['gold_recall']:.1f}%")
print(f"Avg Redundancy: {metrics['avg_redundancy']:.3f}")
print(f"Gold Percentage: {metrics['gold_percentage']:.1f}%")
```

### Option 3: Compare Retrieval Methods

Test QUBO-RAG against baseline methods:

```python
from core.retrieval_strategies import create_retrieval_strategy
from core.retriever import Retriever
from core.vector_store import VectorStore
from core.embedder import EmbeddingGenerator

# Setup
vector_store = VectorStore(
    collection_name="wiki_aspects",
    persist_directory="./data/wikipedia/wikipedia_db"
)

embedder = EmbeddingGenerator(model_name="BAAI/bge-large-en-v1.5")

# Test different strategies
strategies = {
    "Naive Top-K": create_retrieval_strategy("naive"),
    "MMR": create_retrieval_strategy("mmr", lambda_param=0.5),
    "QUBO-RAG": create_retrieval_strategy("qubo", alpha=0.7, solver_preset="balanced")
}

query = "What are the key events in ancient Roman history?"
query_embedding = embedder.embed_query(query)

for name, strategy in strategies.items():
    retriever = Retriever(vector_store, embedder, strategy)
    results = retriever.retrieve(query, k=10)
    print(f"\n{name}:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['text'][:100]}... (score: {result['score']:.3f})")
```

## Dataset Metadata Schema

Each chunk in the Wikipedia ChromaDB has the following metadata:

```python
{
    "chunk_type": str,        # "gold_base", "gold_redundant", "noise", or "prompt"
    "prompt_id": str,         # UUID linking to parent prompt
    "article_title": str,     # Wikipedia article title
    "aspect_id": int,         # 0-4 for gold chunks, -1 for noise/prompt
    "aspect_name": str,       # Aspect description or "noise"/"prompt"
    "redundancy_index": int,  # 0 = base, 1-N = redundant, -1 = noise/prompt
}
```

### Filtering Examples

```python
# Get only base gold chunks (no redundancy)
base_chunks = vector_store.get_by_metadata({
    "chunk_type": "gold_base"
})

# Get chunks from a specific aspect
aspect_chunks = vector_store.get_by_metadata({
    "aspect_id": "0",
    "chunk_type": {"$in": ["gold_base", "gold_redundant"]}
})

# Search excluding high redundancy
results = vector_store.search_with_filters(
    query_embedding,
    k=20,
    metadata_filter={"redundancy_index": {"$lte": "1"}},
    exclude_metadata={"chunk_type": "noise"}
)
```

## Experiments to Run

### 1. Redundancy Handling Comparison
Compare how different retrieval methods handle redundant information:
- Vary redundancy levels (0, 1, 2, 3, 5)
- Measure gold recall and intra-list similarity
- Compare QUBO vs MMR vs Naive

### 2. QUBO Parameter Tuning
Test different alpha values for relevance/diversity tradeoff:
- α = 0.3 (high diversity)
- α = 0.5 (balanced)
- α = 0.7 (high relevance)
- α = 0.9 (minimal diversity)

### 3. Solver Comparison
Compare ORBIT solver presets:
- 'fast': Quick convergence (good for demos)
- 'balanced': Default quality/speed tradeoff
- 'quality': Higher quality solutions
- 'maximum': Best quality, slowest

### 4. Energy Landscape Analysis
Use the precomputed similarity matrices to analyze QUBO energy landscapes and understand optimization behavior.

## Important Notes

### Embedding Model Consistency
**CRITICAL**: You must use `BAAI/bge-large-en-v1.5` for queries since the dataset was embedded with this model. Using a different model (e.g., `all-MiniLM-L6-v2`) will produce incorrect similarity scores.

```python
# CORRECT
embedder = EmbeddingGenerator(model_name="BAAI/bge-large-en-v1.5")

# WRONG - will give bad results
embedder = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
```

### Dataset Size
- Total chunks: 5,600
- ChromaDB size: ~53 MB
- Similarity matrix: ~239 MB
- Cache: Varies (depends on articles cached)

## Next Steps

1. **Run Experiments**: Use the integrated dataset to test QUBO-RAG performance
2. **Visualize Results**: Create plots comparing retrieval methods
3. **Tune Parameters**: Find optimal α and solver settings for your use case
4. **Create Demo Notebook**: Build an interactive demo showcasing diversity-aware retrieval

## Questions or Issues?

- Check the original Quantum-RAG documentation in `docs/`
- Review the Quantum Dice v2 dataset creation pipeline if you need to regenerate data
- Ensure the virtual environment from Quantum-RAG is active (contains all required dependencies)
