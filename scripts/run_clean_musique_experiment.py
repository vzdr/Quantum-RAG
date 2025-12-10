"""
Experiment 1.2: The Clean Control Test

Purpose: Prove QUBO doesn't harm performance on clean data (no poison).
         This establishes that QUBO's diversity optimization doesn't hurt
         when redundancy isn't a problem.

Hypothesis:
- All methods (Top-K, MMR, QUBO) should perform similarly on clean data
- QUBO should be within ±5% of Top-K on chain recall
- This proves QUBO is a "safe" replacement that only helps, never hurts

Dataset: Original MuSiQue dev set (no poison/clones)
- 100 questions with 3-hop reasoning chains
- NO semantic clones (only gold facts + distractors)
- Total candidate pool: ~13 paragraphs per question (3 gold + 10 distractors)

Comparison: Top-K vs MMR vs QUBO

Metrics:
- Chain Recall: Percentage of questions where ALL gold facts retrieved
- Gold Recall: Percentage of individual gold facts retrieved
- Redundancy Score: Average pairwise cosine similarity in top-K

Usage:
    python scripts/run_clean_musique_experiment.py
    python scripts/run_clean_musique_experiment.py --n-questions 50 --k 5
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Set
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.retrieval_strategies import create_retrieval_strategy

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Error: sentence-transformers not installed")
    print("Install with: pip install sentence-transformers")
    sys.exit(1)


@dataclass
class GoldFact:
    """A gold fact required for answering the question."""
    id: str
    text: str
    embedding: np.ndarray = None


@dataclass
class Question:
    """A multi-hop question with gold facts (no poison)."""
    id: str
    query: str
    gold_facts: List[GoldFact]
    distractor_paragraphs: List[Dict]


class CleanDatasetGenerator:
    """
    Generates a clean dataset for testing retrieval on non-redundant data.

    Unlike the poisoned dataset, this has no semantic clones.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        print("Loading embedding model...")
        self.model = SentenceTransformer(model_name)

    def generate_synthetic_dataset(
        self,
        n_questions: int = 100,
        n_hops: int = 3,
        n_distractors: int = 10
    ) -> List[Question]:
        """
        Generate synthetic multi-hop questions WITHOUT poison clones.

        Each question has:
        - n_hops gold facts (required for answer)
        - n_distractors irrelevant paragraphs
        - NO clones
        """
        print(f"Generating {n_questions} CLEAN synthetic questions (no poison)...")

        # Topic templates for diverse questions
        topics = [
            "history", "science", "geography", "culture", "technology",
            "medicine", "economics", "literature", "sports", "music"
        ]

        questions = []

        for q_idx in range(n_questions):
            topic_name = topics[q_idx % len(topics)]

            # Generate gold facts for this question (simulating multi-hop chain)
            gold_facts = []
            for hop in range(n_hops):
                # Each gold fact has UNIQUE content to ensure they're distinct
                fact_text = f"[GOLD-{q_idx}-{hop}] Gold fact {hop+1} for question {q_idx} " \
                           f"about {topic_name}. This is hop {hop+1} of {n_hops}. " \
                           f"Unique identifier: HOP{hop}_Q{q_idx}_{topic_name.upper()}."

                gold_facts.append(GoldFact(
                    id=f"gold_{q_idx}_{hop}",
                    text=fact_text
                ))

            # Generate distractor paragraphs - DIVERSE topics
            distractors = []
            distractor_topics = ["astronomy", "cooking", "politics", "fashion",
                                "architecture", "biology", "chemistry", "physics",
                                "philosophy", "religion"]

            for d_idx in range(n_distractors):
                d_topic = distractor_topics[d_idx % len(distractor_topics)]
                distractor_text = f"[DISTRACTOR-{q_idx}-{d_idx}] This paragraph discusses " \
                                 f"{d_topic}, which is unrelated to {topic_name}. " \
                                 f"Content about {d_topic} topic number {d_idx}."

                distractors.append({
                    'id': f"distractor_{q_idx}_{d_idx}",
                    'text': distractor_text,
                    'is_gold': False
                })

            # Create query
            query = f"Multi-hop question {q_idx}: What is the complete chain of facts " \
                   f"about {topic_name} involving all {n_hops} reasoning steps?"

            questions.append(Question(
                id=f"q_{q_idx}",
                query=query,
                gold_facts=gold_facts,
                distractor_paragraphs=distractors
            ))

        # Compute embeddings
        print("Computing embeddings for all texts...")
        self._compute_embeddings(questions)

        return questions

    def _compute_embeddings(self, questions: List[Question]):
        """Compute embeddings for all texts."""
        all_texts = []
        text_mapping = []

        for q_idx, question in enumerate(questions):
            # Query
            all_texts.append(question.query)
            text_mapping.append((q_idx, 'query', 0))

            # Gold facts
            for g_idx, gold in enumerate(question.gold_facts):
                all_texts.append(gold.text)
                text_mapping.append((q_idx, 'gold', g_idx))

            # Distractors
            for d_idx, distractor in enumerate(question.distractor_paragraphs):
                all_texts.append(distractor['text'])
                text_mapping.append((q_idx, 'distractor', d_idx))

        # Batch encode
        embeddings = self.model.encode(all_texts, show_progress_bar=True, convert_to_numpy=True)

        # Assign embeddings back
        for (q_idx, item_type, item_idx), embedding in zip(text_mapping, embeddings):
            question = questions[q_idx]

            if item_type == 'query':
                question.query_embedding = embedding
            elif item_type == 'gold':
                question.gold_facts[item_idx].embedding = embedding
            elif item_type == 'distractor':
                question.distractor_paragraphs[item_idx]['embedding'] = embedding


def build_candidate_pool(question: Question) -> List[Dict]:
    """Build candidate pool for retrieval from a question (no clones)."""
    candidates = []

    # Add gold facts
    for gold in question.gold_facts:
        candidates.append({
            'id': gold.id,
            'text': gold.text,
            'embedding': gold.embedding,
            'metadata': {'source': gold.id, 'chunk_index': 0, 'start_char': 0, 'end_char': len(gold.text)},
            'is_gold': True
        })

    # Add distractors
    for distractor in question.distractor_paragraphs:
        candidates.append({
            'id': distractor['id'],
            'text': distractor['text'],
            'embedding': distractor['embedding'],
            'metadata': {'source': distractor['id'], 'chunk_index': 0, 'start_char': 0, 'end_char': len(distractor['text'])},
            'is_gold': False
        })

    return candidates


def compute_query_similarities(query_embedding: np.ndarray, candidates: List[Dict]) -> List[Dict]:
    """Compute similarity scores between query and candidates."""
    candidate_embeddings = np.array([c['embedding'] for c in candidates])

    # Normalize
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    candidate_norms = candidate_embeddings / np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)

    # Compute similarities
    similarities = np.dot(candidate_norms, query_norm)

    # Add scores to candidates
    for candidate, sim in zip(candidates, similarities):
        candidate['score'] = float(sim)

    # Sort by score (descending)
    candidates.sort(key=lambda x: x['score'], reverse=True)

    return candidates


def evaluate_retrieval(retrieved_ids: Set[str], question: Question) -> Dict[str, float]:
    """Evaluate retrieval results."""
    gold_ids = {g.id for g in question.gold_facts}

    # Count gold facts retrieved
    gold_retrieved = retrieved_ids & gold_ids
    gold_recall = len(gold_retrieved) / len(gold_ids) if gold_ids else 0

    # Chain recall: 1 only if ALL gold facts retrieved
    chain_recall = 1.0 if gold_retrieved == gold_ids else 0.0

    return {
        'chain_recall': chain_recall,
        'gold_recall': gold_recall,
        'n_gold_retrieved': len(gold_retrieved)
    }


def compute_redundancy_score(embeddings: List[np.ndarray]) -> float:
    """Compute average pairwise similarity (redundancy) of embeddings."""
    if len(embeddings) < 2:
        return 0.0

    embeddings_array = np.array(embeddings)
    norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    normalized = embeddings_array / norms

    sim_matrix = np.dot(normalized, normalized.T)

    n = len(embeddings)
    upper_tri_sum = np.sum(np.triu(sim_matrix, k=1))
    n_pairs = n * (n - 1) / 2

    return upper_tri_sum / n_pairs if n_pairs > 0 else 0.0


def run_experiment(
    questions: List[Question],
    k: int = 5,
    alpha: float = 0.7,
    lambda_param: float = 0.5
) -> Dict[str, Dict[str, float]]:
    """
    Run the clean control test experiment.

    Compares Top-K, MMR, and QUBO on clean data (no poison).
    """
    print(f"\nRunning CLEAN experiment with k={k}, alpha={alpha}, lambda={lambda_param}")
    print(f"Number of questions: {len(questions)}")

    # Create strategies
    strategies = {
        'Top-K': create_retrieval_strategy('naive'),
        'MMR': create_retrieval_strategy('mmr', lambda_param=lambda_param),
        'QUBO': create_retrieval_strategy('qubo', alpha=alpha, solver_preset='balanced')
    }

    # Results storage
    results = {name: defaultdict(list) for name in strategies.keys()}

    for q_idx, question in enumerate(questions):
        if (q_idx + 1) % 10 == 0:
            print(f"  Processing question {q_idx + 1}/{len(questions)}...")

        # Build candidate pool (no clones)
        candidates = build_candidate_pool(question)

        # Compute query similarities
        candidates = compute_query_similarities(question.query_embedding, candidates)

        # Run each strategy
        for strategy_name, strategy in strategies.items():
            # Retrieve
            retrieved_results, metadata = strategy.retrieve(
                query_embedding=question.query_embedding,
                candidate_results=candidates,
                k=k
            )

            # Get retrieved IDs and embeddings
            retrieved_ids = {r.id for r in retrieved_results}
            retrieved_embeddings = []
            for r in retrieved_results:
                for c in candidates:
                    if c['id'] == r.id:
                        retrieved_embeddings.append(c['embedding'])
                        break

            # Evaluate
            eval_metrics = evaluate_retrieval(retrieved_ids, question)

            # Compute redundancy
            redundancy = compute_redundancy_score(retrieved_embeddings)

            # Store results
            results[strategy_name]['chain_recall'].append(eval_metrics['chain_recall'])
            results[strategy_name]['gold_recall'].append(eval_metrics['gold_recall'])
            results[strategy_name]['redundancy'].append(redundancy)

    # Aggregate results
    aggregated = {}
    for strategy_name, metrics in results.items():
        aggregated[strategy_name] = {
            'chain_recall': np.mean(metrics['chain_recall']),
            'chain_recall_std': np.std(metrics['chain_recall']),
            'gold_recall': np.mean(metrics['gold_recall']),
            'gold_recall_std': np.std(metrics['gold_recall']),
            'redundancy': np.mean(metrics['redundancy']),
            'redundancy_std': np.std(metrics['redundancy'])
        }

    return aggregated


def print_results(results: Dict[str, Dict[str, float]]):
    """Print experiment results in a formatted table."""
    print("\n" + "="*80)
    print("EXPERIMENT 1.2: CLEAN CONTROL TEST RESULTS")
    print("="*80)

    print(f"\n{'Metric':<20} {'Top-K':<20} {'MMR':<20} {'QUBO':<20}")
    print("-"*80)

    metrics = ['chain_recall', 'gold_recall', 'redundancy']
    metric_names = ['Chain Recall', 'Gold Recall', 'Redundancy']

    for metric, name in zip(metrics, metric_names):
        row = f"{name:<20}"
        for strategy in ['Top-K', 'MMR', 'QUBO']:
            mean = results[strategy][metric]
            std = results[strategy][f'{metric}_std']
            row += f"{mean:.3f} ± {std:.3f}      "
        print(row)

    print("="*80)

    # Check if QUBO is within ±5% of Top-K
    topk_cr = results['Top-K']['chain_recall']
    qubo_cr = results['QUBO']['chain_recall']
    diff = abs(qubo_cr - topk_cr)

    print(f"\nQUBO vs Top-K Chain Recall difference: {diff:.3f} ({diff/topk_cr*100:.1f}%)")
    if diff <= 0.05:
        print("✓ PASS: QUBO is within ±5% of Top-K (no harm on clean data)")
    else:
        print("✗ FAIL: QUBO differs from Top-K by more than 5%")

    print("="*80)


def plot_results(results: Dict[str, Dict[str, float]], output_file: str):
    """Create bar chart visualization of results."""
    metrics = ['chain_recall', 'gold_recall', 'redundancy']
    metric_names = ['Chain Recall', 'Gold Recall', 'Redundancy']
    strategies = ['Top-K', 'MMR', 'QUBO']
    colors = ['#3498db', '#3498db', '#3498db']  # All blue (similar performance expected)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    x = np.arange(len(strategies))
    width = 0.6

    for ax, metric, name in zip(axes, metrics, metric_names):
        values = [results[s][metric] for s in strategies]
        errors = [results[s][f'{metric}_std'] for s in strategies]

        bars = ax.bar(x, values, width, yerr=errors, capsize=5, color=colors)

        ax.set_ylabel(name)
        ax.set_title(name)
        ax.set_xticks(x)
        ax.set_xticklabels(strategies)
        ax.set_ylim(0, 1.1)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=10)

    plt.suptitle('Experiment 1.2: Clean Control Test - Top-K vs MMR vs QUBO',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Run Experiment 1.2: Clean Control Test')
    parser.add_argument('--n-questions', type=int, default=100, help='Number of questions')
    parser.add_argument('--k', type=int, default=5, help='Number of results to retrieve')
    parser.add_argument('--n-hops', type=int, default=3, help='Number of hops (gold facts) per question')
    parser.add_argument('--alpha', type=float, default=0.7, help='QUBO alpha parameter')
    parser.add_argument('--lambda-param', type=float, default=0.5, help='MMR lambda parameter')
    parser.add_argument('--output-dir', type=str, default='experiments/results', help='Output directory')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate CLEAN dataset (no poison)
    generator = CleanDatasetGenerator()
    questions = generator.generate_synthetic_dataset(
        n_questions=args.n_questions,
        n_hops=args.n_hops
    )

    # Run experiment
    results = run_experiment(
        questions=questions,
        k=args.k,
        alpha=args.alpha,
        lambda_param=args.lambda_param
    )

    # Print results
    print_results(results)

    # Save results
    results_file = output_dir / 'exp_1_2_clean_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_file}")

    # Plot results
    plot_file = output_dir / 'exp_1_2_clean_chart.png'
    plot_results(results, str(plot_file))


if __name__ == "__main__":
    main()
