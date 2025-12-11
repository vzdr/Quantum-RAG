"""
Greedy Trap Benchmark: Demonstrating QUBO's superiority over MMR

Dataset designed with documents that are lexically diverse but semantically similar.
MMR's cosine-based diversity is fooled, QUBO's global optimization is not.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
from collections import defaultdict

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class GreedyTrapBenchmark:
    def __init__(self, data_dir=None):
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "data" / "greedy_trap"
        self.data_dir = Path(data_dir)

        print("Loading embedding model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        self.documents = []
        self.clusters = []
        self.filenames = []
        self._load_documents()

        print(f"Computing embeddings for {len(self.documents)} documents...")
        self.embeddings = self.model.encode(self.documents, show_progress_bar=True)

    def _load_documents(self):
        cluster_prefixes = ['fatigue', 'joint', 'fever', 'muscle', 'cognitive']
        for filepath in sorted(self.data_dir.glob("*.txt")):
            with open(filepath, 'r', encoding='utf-8') as f:
                self.documents.append(f.read().strip())
            self.filenames.append(filepath.name)
            for prefix in cluster_prefixes:
                if filepath.name.startswith(prefix):
                    self.clusters.append(prefix)
                    break
            else:
                self.clusters.append('unknown')

        print(f"Loaded {len(self.documents)} documents")

    def top_k_retrieval(self, query_embedding, k=5):
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        return list(np.argsort(similarities)[::-1][:k])

    def mmr_retrieval(self, query_embedding, k=5, lambda_param=0.5):
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        doc_similarities = cosine_similarity(self.embeddings)

        selected = []
        candidates = list(range(len(self.documents)))

        for _ in range(k):
            if not candidates:
                break
            mmr_scores = []
            for idx in candidates:
                relevance = similarities[idx]
                max_sim = max((doc_similarities[idx][s] for s in selected), default=0)
                mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
                mmr_scores.append((idx, mmr))
            best_idx = max(mmr_scores, key=lambda x: x[1])[0]
            selected.append(best_idx)
            candidates.remove(best_idx)

        return selected

    def qubo_retrieval(self, query_embedding, k=5, alpha=0.5):
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        doc_similarities = cosine_similarity(self.embeddings)
        n = len(self.documents)

        Q = np.zeros((n, n))
        for i in range(n):
            Q[i, i] = -alpha * similarities[i]
            for j in range(i + 1, n):
                penalty = (1 - alpha) * doc_similarities[i, j]
                Q[i, j] = penalty
                Q[j, i] = penalty

        # Simulated annealing solver
        relevances = -np.diag(Q)
        current = set(np.argsort(relevances)[::-1][:k])
        current_energy = sum(Q[i, i] + sum(Q[i, j] for j in current if j > i) for i in current)
        best = current.copy()
        best_energy = current_energy

        T = 1.0
        for _ in range(5000):
            remove = np.random.choice(list(current))
            add = np.random.choice([i for i in range(n) if i not in current])
            new = current.copy()
            new.remove(remove)
            new.add(add)
            new_energy = sum(Q[i, i] + sum(Q[i, j] for j in new if j > i) for i in new)

            if new_energy < current_energy or np.random.random() < np.exp((current_energy - new_energy) / T):
                current = new
                current_energy = new_energy
                if current_energy < best_energy:
                    best = current.copy()
                    best_energy = current_energy
            T *= 0.995

        return list(best)

    def run_benchmark(self, query="I feel tired all the time", k=5):
        print(f"\nQuery: '{query}'\n")
        query_emb = self.model.encode([query])[0]

        for name, method in [('Top-K', self.top_k_retrieval),
                             ('MMR', self.mmr_retrieval),
                             ('QUBO', self.qubo_retrieval)]:
            indices = method(query_emb, k)
            clusters = set(self.clusters[i] for i in indices)
            print(f"{name:8} Coverage: {len(clusters)}/5 {clusters}")
            for i in indices:
                print(f"         {self.filenames[i]}")


if __name__ == "__main__":
    benchmark = GreedyTrapBenchmark()
    benchmark.run_benchmark()
