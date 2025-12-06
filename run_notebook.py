#!/usr/bin/env python
"""
Run the RAG System notebook from command line.

Usage:
    python run_notebook.py                    # Run full notebook
    python run_notebook.py --phase A          # Run Phase A only (indexing)
    python run_notebook.py --phase B          # Run Phase B only (query)
    python run_notebook.py --phase C          # Run Phase C only (diversity comparison)
    python run_notebook.py --query "Who is Frodo?"  # Quick query
    python run_notebook.py --compare "What is the One Ring?"  # Compare methods
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_full_notebook():
    """Execute the full notebook."""
    import nbformat
    from nbclient import NotebookClient

    print("=" * 70)
    print("Running RAG System Notebook")
    print("=" * 70)

    # Load notebook
    with open('RAG_System.ipynb', 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # Create client with quantum-rag kernel
    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name='quantum-rag',
        resources={'metadata': {'path': '.'}}
    )

    # Execute
    try:
        client.execute()
        print("\nNotebook executed successfully!")
    except Exception as e:
        print(f"\nError executing notebook: {e}")
        raise

    # Save executed notebook
    with open('RAG_System_executed.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print("Saved executed notebook to: RAG_System_executed.ipynb")


def run_indexing(data_path='./data/lotr'):
    """Run Phase A: Load, chunk, embed, store."""
    import warnings
    warnings.filterwarnings('ignore')

    from dotenv import load_dotenv
    load_dotenv()

    from core import DocumentLoader, TextChunker, EmbeddingGenerator, VectorStore
    from config import RAGConfig

    print("=" * 70)
    print("Phase A: Indexing")
    print("=" * 70)

    config = RAGConfig()

    # Load documents
    print(f"\nLoading documents from {data_path}...")
    documents = DocumentLoader.load_directory(data_path)
    for doc in documents:
        print(f"  {doc.source}: {len(doc):,} chars")
    print(f"Total: {len(documents)} docs")

    # Chunk
    print("\nChunking...")
    chunker = TextChunker(chunk_size=config.chunk_size, overlap=config.chunk_overlap, strategy=config.chunking_strategy)
    chunks = chunker.chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")

    # Embed
    print("\nGenerating embeddings...")
    embedder = EmbeddingGenerator(model_name=config.embedding_model)
    embedded_chunks = embedder.embed_chunks(chunks, batch_size=32, show_progress=True)
    print(f"Generated {len(embedded_chunks)} embeddings")

    # Store
    print("\nStoring in vector database...")
    vector_store = VectorStore(collection_name='rag_collection', persist_directory='./chroma_db', reset=True)
    count = vector_store.add(embedded_chunks)
    print(f"Stored {count} vectors")

    print("\nIndexing complete!")
    return embedder, vector_store


def run_query(query, embedder=None, vector_store=None, method='mmr', k=5):
    """Run a query with specified retrieval method."""
    import warnings
    warnings.filterwarnings('ignore')

    from dotenv import load_dotenv
    load_dotenv()

    from core import EmbeddingGenerator, VectorStore, ResponseGenerator
    from core.retrieval_strategies import create_retrieval_strategy
    from config import RAGConfig

    config = RAGConfig()

    # Initialize components if not provided
    if embedder is None:
        print("Loading embedder...")
        embedder = EmbeddingGenerator(model_name=config.embedding_model)

    if vector_store is None:
        print("Loading vector store...")
        vector_store = VectorStore(collection_name='rag_collection', persist_directory='./chroma_db')

    # Create strategy
    if method == 'qubo':
        try:
            import orbit
            strategy = create_retrieval_strategy('qubo', alpha=0.6,
                                                  solver_params={'n_replicas': 2, 'full_sweeps': 5000})
        except ImportError:
            print("ORBIT not available, falling back to MMR")
            method = 'mmr'
            strategy = create_retrieval_strategy('mmr', lambda_param=0.5)
    elif method == 'mmr':
        strategy = create_retrieval_strategy('mmr', lambda_param=0.5)
    else:
        strategy = create_retrieval_strategy('naive')

    print("=" * 70)
    print(f"Query: {query}")
    print(f"Method: {method.upper()}")
    print("=" * 70)

    # Retrieve
    query_emb = embedder.embed_query(query)
    candidates = vector_store.search(query_emb, k=k*3)
    results, metadata = strategy.retrieve(query_emb, candidates, k=k)

    print(f"\nRetrieved {len(results)} chunks:")
    for r in results:
        print(f"\n[{r.rank}] Score: {r.score:.3f} | {r.source}")
        print(f"    {r.text[:200]}...")

    # Generate
    print("\nGenerating response...")
    generator = ResponseGenerator(model=config.llm_model)
    response = generator.generate(query=query, context_chunks=results)

    print("\n" + "=" * 70)
    print("ANSWER:")
    print("=" * 70)
    print(response.response)
    print(f"\nSources: {', '.join(response.sources)}")

    return results, response


def run_comparison(query, k=5):
    """Compare all retrieval methods on a query."""
    import warnings
    warnings.filterwarnings('ignore')

    from dotenv import load_dotenv
    load_dotenv()

    import numpy as np
    from core import EmbeddingGenerator, VectorStore
    from core.retrieval_strategies import create_retrieval_strategy
    from core.diversity_metrics import compare_retrieval_methods, print_comparison_table
    from config import RAGConfig

    config = RAGConfig()

    print("Loading components...")
    embedder = EmbeddingGenerator(model_name=config.embedding_model)
    vector_store = VectorStore(collection_name='rag_collection', persist_directory='./chroma_db')

    print("=" * 70)
    print(f"Comparing methods on: {query}")
    print("=" * 70)

    # Get candidates
    query_emb = embedder.embed_query(query)
    candidates = vector_store.search(query_emb, k=k*3)

    # Define strategies
    strategies = {
        'Naive': create_retrieval_strategy('naive'),
        'MMR (lambda=0.5)': create_retrieval_strategy('mmr', lambda_param=0.5),
    }

    # Try to add QUBO
    try:
        import orbit
        strategies['QUBO (alpha=0.6)'] = create_retrieval_strategy('qubo', alpha=0.6,
                                                   solver_params={'n_replicas': 2, 'full_sweeps': 5000})
    except ImportError:
        print("Note: QUBO disabled (requires Python 3.13+ with ORBIT)\n")

    results_dict = {}
    for name, strategy in strategies.items():
        print(f"\nRunning {name}...")
        results, metadata = strategy.retrieve(query_emb, candidates, k=k)

        results_with_emb = []
        for r in results:
            orig = next((c for c in candidates if c['id'] == r.id), None)
            if orig:
                results_with_emb.append({
                    'id': r.id,
                    'score': r.score,
                    'embedding': orig.get('embedding'),
                })
        results_dict[name] = results_with_emb

        exec_time = metadata.get('execution_time', 0)
        print(f"  Retrieved {len(results)} chunks in {exec_time:.3f}s")

    # Compare
    print("\n" + "=" * 70)
    print("DIVERSITY METRICS")
    print("=" * 70)
    comparison = compare_retrieval_methods(results_dict)
    print_comparison_table(comparison)

    return results_dict, comparison


def main():
    parser = argparse.ArgumentParser(description='Run RAG System notebook from command line')
    parser.add_argument('--phase', choices=['A', 'B', 'C', 'all'], default='all',
                        help='Which phase to run (A=indexing, B=query, C=diversity)')
    parser.add_argument('--query', type=str, help='Run a quick query')
    parser.add_argument('--compare', type=str, help='Compare retrieval methods on a query')
    parser.add_argument('--method', choices=['naive', 'mmr', 'qubo'], default='mmr',
                        help='Retrieval method for --query')
    parser.add_argument('--data', type=str, default='./data/lotr',
                        help='Data directory for indexing')
    parser.add_argument('--k', type=int, default=5, help='Number of chunks to retrieve')
    parser.add_argument('--notebook', action='store_true', help='Execute full notebook')
    parser.add_argument('--skip-index', action='store_true', help='Skip indexing, use existing DB')

    args = parser.parse_args()

    if args.notebook:
        run_full_notebook()
    elif args.query:
        run_query(args.query, method=args.method, k=args.k)
    elif args.compare:
        run_comparison(args.compare, k=args.k)
    elif args.phase == 'A':
        run_indexing(args.data)
    elif args.phase == 'B':
        run_query("What is the One Ring?", method=args.method, k=args.k)
    elif args.phase == 'C':
        run_comparison("Who has the One Ring and what is its power?", k=args.k)
    else:
        # Run all phases
        embedder, vector_store = run_indexing(args.data)
        print("\n")
        run_query("What is the One Ring?", embedder=embedder, vector_store=vector_store, method=args.method, k=args.k)
        print("\n")
        run_comparison("Who has the One Ring and what is its power?", k=args.k)


if __name__ == '__main__':
    main()
