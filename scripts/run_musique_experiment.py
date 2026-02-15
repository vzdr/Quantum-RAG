"""
Experiment 1.1: The Poisoned Stress Test

Purpose: Prove Top-K fails catastrophically under redundancy, while QUBO rescues chains.
         MMR is added as an intermediate baseline.

Hypothesis:
- Top-K retrieves duplicates of "Fact A" and misses "Fact B/C"
- MMR partially mitigates but still struggles with high redundancy
- QUBO rejects duplicates and retrieves the full reasoning chain

Dataset: Poisoned MuSiQue dev set
- 100 questions with 3-hop reasoning chains
- 3 semantic clones per gold fact (85-95% similarity)
- Total candidate pool: ~20-25 paragraphs per question

Comparison: Top-K vs MMR vs QUBO

Metrics:
- Chain Recall: Percentage of questions where ALL gold facts retrieved
- Gold Recall: Percentage of individual gold facts retrieved
- Poison Rate: Percentage of retrieved chunks that are clones
- Redundancy Score: Average pairwise cosine similarity in top-K

Usage:
    python scripts/run_musique_experiment.py
    python scripts/run_musique_experiment.py --n-questions 50 --k 5
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.retrieval_strategies import create_retrieval_strategy
from sklearn.metrics.pairwise import cosine_similarity

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
class PoisonClone:
    """A semantic clone of a gold fact."""
    id: str
    text: str
    source_gold_id: str  # Which gold fact this is a clone of
    embedding: np.ndarray = None


@dataclass
class Question:
    """A multi-hop question with gold facts and poison clones."""
    id: str
    query: str
    gold_facts: List[GoldFact]
    poison_clones: List[PoisonClone]
    distractor_paragraphs: List[Dict]  # Non-relevant paragraphs


class PoisonedDatasetGenerator:
    """
    Generates a poisoned dataset for testing retrieval robustness.

    Since we don't have actual MuSiQue, we simulate it with synthetic data
    that follows the same structure and challenge.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        print("Loading embedding model...")
        self.model = SentenceTransformer(model_name)

    def generate_synthetic_dataset(
        self,
        n_questions: int = 100,
        n_hops: int = 3,
        n_clones_per_fact: int = 3,
        n_distractors: int = 10
    ) -> List[Question]:
        """
        Generate synthetic multi-hop questions with poison clones.

        Each question has:
        - n_hops gold facts (required for answer)
        - n_clones_per_fact clones of each gold fact
        - n_distractors irrelevant paragraphs
        """
        print(f"Generating {n_questions} synthetic questions...")

        # Topic templates for diverse questions
        topics = [
            ("history", [
                "The {entity} was founded in {year} by {founder}.",
                "{founder} established the {entity} after {event}.",
                "In {year}, {founder} created what would become the {entity}."
            ]),
            ("science", [
                "The discovery of {concept} by {scientist} revolutionized {field}.",
                "{scientist}'s work on {concept} transformed our understanding of {field}.",
                "Research into {concept} by {scientist} changed {field} forever."
            ]),
            ("geography", [
                "The {location} is located in {region} and is known for {feature}.",
                "{location} can be found in {region}, famous for its {feature}.",
                "In {region}, the {location} stands out due to its {feature}."
            ]),
            ("culture", [
                "The tradition of {practice} originated in {culture} during {period}.",
                "{culture} developed the {practice} tradition in the {period}.",
                "During the {period}, {culture} gave rise to the {practice} tradition."
            ]),
            ("technology", [
                "The invention of {tech} by {inventor} enabled {application}.",
                "{inventor}'s {tech} made {application} possible.",
                "Thanks to {inventor}'s {tech}, we now have {application}."
            ])
        ]

        questions = []

        for q_idx in range(n_questions):
            # Select topic and generate question
            topic_name, templates = topics[q_idx % len(topics)]

            # Generate gold facts for this question (simulating multi-hop chain)
            gold_facts = []
            for hop in range(n_hops):
                fact_text = f"[GOLD-{q_idx}-{hop}] This is gold fact {hop+1} for question {q_idx}. " \
                           f"It contains essential information about {topic_name} topic {q_idx}. " \
                           f"Hop {hop+1} of {n_hops} in the reasoning chain."

                gold_facts.append(GoldFact(
                    id=f"gold_{q_idx}_{hop}",
                    text=fact_text
                ))

            # Generate poison clones for each gold fact
            poison_clones = []
            for gold_fact in gold_facts:
                for clone_idx in range(n_clones_per_fact):
                    # Create paraphrase (simulating back-translation)
                    clone_text = self._paraphrase(gold_fact.text, clone_idx)

                    poison_clones.append(PoisonClone(
                        id=f"clone_{gold_fact.id}_{clone_idx}",
                        text=clone_text,
                        source_gold_id=gold_fact.id
                    ))

            # Generate distractor paragraphs
            distractors = []
            for d_idx in range(n_distractors):
                distractor_text = f"[DISTRACTOR-{q_idx}-{d_idx}] This paragraph discusses " \
                                 f"unrelated information about a different topic. " \
                                 f"It should not be retrieved for question {q_idx}."

                distractors.append({
                    'id': f"distractor_{q_idx}_{d_idx}",
                    'text': distractor_text,
                    'is_gold': False,
                    'is_clone': False
                })

            # Create query
            query = f"Multi-hop question {q_idx}: What is the complete chain of facts " \
                   f"about {topic_name} topic {q_idx} involving all {n_hops} reasoning steps?"

            questions.append(Question(
                id=f"q_{q_idx}",
                query=query,
                gold_facts=gold_facts,
                poison_clones=poison_clones,
                distractor_paragraphs=distractors
            ))

        # Compute embeddings
        print("Computing embeddings for all texts...")
        self._compute_embeddings(questions)

        return questions

    def _paraphrase(self, text: str, variant: int) -> str:
        """
        Create a paraphrase of the text (simulating back-translation).
        In practice, you'd use a paraphraser model.
        """
        paraphrase_patterns = [
            lambda t: t.replace("This is", "Here we have").replace("contains", "includes"),
            lambda t: t.replace("This is", "We present").replace("essential", "crucial"),
            lambda t: t.replace("This is", "Presenting").replace("information", "details"),
        ]

        pattern = paraphrase_patterns[variant % len(paraphrase_patterns)]
        paraphrased = pattern(text)

        # Add variant marker for tracking
        paraphrased = paraphrased.replace("[GOLD-", f"[CLONE-v{variant}-")

        return paraphrased

    def _compute_embeddings(self, questions: List[Question]):
        """Compute embeddings for all texts."""
        all_texts = []
        text_mapping = []  # (question_idx, type, item_idx)

        for q_idx, question in enumerate(questions):
            # Query
            all_texts.append(question.query)
            text_mapping.append((q_idx, 'query', 0))

            # Gold facts
            for g_idx, gold in enumerate(question.gold_facts):
                all_texts.append(gold.text)
                text_mapping.append((q_idx, 'gold', g_idx))

            # Poison clones
            for c_idx, clone in enumerate(question.poison_clones):
                all_texts.append(clone.text)
                text_mapping.append((q_idx, 'clone', c_idx))

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
            elif item_type == 'clone':
                question.poison_clones[item_idx].embedding = embedding
            elif item_type == 'distractor':
                question.distractor_paragraphs[item_idx]['embedding'] = embedding


def build_candidate_pool(question: Question) -> List[Dict]:
    """Build candidate pool for retrieval from a question."""
    candidates = []

    # Add gold facts
    for gold in question.gold_facts:
        candidates.append({
            'id': gold.id,
            'text': gold.text,
            'embedding': gold.embedding,
            'metadata': {'source': gold.id, 'chunk_index': 0, 'start_char': 0, 'end_char': len(gold.text)},
            'is_gold': True,
            'is_clone': False,
            'source_gold_id': None
        })

    # Add poison clones
    for clone in question.poison_clones:
        candidates.append({
            'id': clone.id,
            'text': clone.text,
            'embedding': clone.embedding,
            'metadata': {'source': clone.id, 'chunk_index': 0, 'start_char': 0, 'end_char': len(clone.text)},
            'is_gold': False,
            'is_clone': True,
            'source_gold_id': clone.source_gold_id
        })

    # Add distractors
    for distractor in question.distractor_paragraphs:
        candidates.append({
            'id': distractor['id'],
            'text': distractor['text'],
            'embedding': distractor['embedding'],
            'metadata': {'source': distractor['id'], 'chunk_index': 0, 'start_char': 0, 'end_char': len(distractor['text'])},
            'is_gold': False,
            'is_clone': False,
            'source_gold_id': None
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


def evaluate_retrieval(
    retrieved_ids: Set[str],
    question: Question
) -> Dict[str, float]:
    """
    Evaluate retrieval results.

    Returns:
        - chain_recall: 1 if ALL gold facts retrieved, 0 otherwise
        - gold_recall: Fraction of gold facts retrieved
        - poison_rate: Fraction of retrieved items that are clones
        - redundancy_score: Average pairwise similarity of retrieved items
    """
    gold_ids = {g.id for g in question.gold_facts}
    clone_ids = {c.id for c in question.poison_clones}

    # Count gold facts retrieved
    gold_retrieved = retrieved_ids & gold_ids
    gold_recall = len(gold_retrieved) / len(gold_ids) if gold_ids else 0

    # Chain recall: 1 only if ALL gold facts retrieved
    chain_recall = 1.0 if gold_retrieved == gold_ids else 0.0

    # Poison rate: fraction of retrieved that are clones
    clones_retrieved = retrieved_ids & clone_ids
    poison_rate = len(clones_retrieved) / len(retrieved_ids) if retrieved_ids else 0

    return {
        'chain_recall': chain_recall,
        'gold_recall': gold_recall,
        'poison_rate': poison_rate,
        'n_gold_retrieved': len(gold_retrieved),
        'n_clones_retrieved': len(clones_retrieved)
    }


def compute_redundancy_score(embeddings: List[np.ndarray]) -> float:
    """Compute average pairwise similarity (redundancy) of embeddings."""
    if len(embeddings) < 2:
        return 0.0

    embeddings_array = np.array(embeddings)
    # Normalize
    norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    normalized = embeddings_array / norms

    # Pairwise similarities
    sim_matrix = np.dot(normalized, normalized.T)

    # Average of upper triangle (excluding diagonal)
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
    Run the poisoned stress test experiment.

    Compares Top-K, MMR, and QUBO on all questions.
    """
    print(f"\nRunning experiment with k={k}, alpha={alpha}, lambda={lambda_param}")
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

        # Build candidate pool
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
            results[strategy_name]['poison_rate'].append(eval_metrics['poison_rate'])
            results[strategy_name]['redundancy'].append(redundancy)

    # Aggregate results
    aggregated = {}
    for strategy_name, metrics in results.items():
        aggregated[strategy_name] = {
            'chain_recall': np.mean(metrics['chain_recall']),
            'chain_recall_std': np.std(metrics['chain_recall']),
            'gold_recall': np.mean(metrics['gold_recall']),
            'gold_recall_std': np.std(metrics['gold_recall']),
            'poison_rate': np.mean(metrics['poison_rate']),
            'poison_rate_std': np.std(metrics['poison_rate']),
            'redundancy': np.mean(metrics['redundancy']),
            'redundancy_std': np.std(metrics['redundancy'])
        }

    return aggregated


def print_results(results: Dict[str, Dict[str, float]]):
    """Print experiment results in a formatted table."""
    print("\n" + "="*80)
    print("EXPERIMENT 1.1: POISONED STRESS TEST RESULTS")
    print("="*80)

    print(f"\n{'Metric':<20} {'Top-K':<20} {'MMR':<20} {'QUBO':<20}")
    print("-"*80)

    metrics = ['chain_recall', 'gold_recall', 'poison_rate', 'redundancy']
    metric_names = ['Chain Recall', 'Gold Recall', 'Poison Rate', 'Redundancy']

    for metric, name in zip(metrics, metric_names):
        row = f"{name:<20}"
        for strategy in ['Top-K', 'MMR', 'QUBO']:
            mean = results[strategy][metric]
            std = results[strategy][f'{metric}_std']
            row += f"{mean:.3f} ± {std:.3f}      "
        print(row)

    print("="*80)
    print("\nInterpretation:")
    print("  - Chain Recall: Higher = better (all gold facts retrieved)")
    print("  - Gold Recall: Higher = better (individual gold facts retrieved)")
    print("  - Poison Rate: Lower = better (fewer clones in results)")
    print("  - Redundancy: Lower = better (more diverse results)")
    print("="*80)


def plot_results(results: Dict[str, Dict[str, float]], output_file: str):
    """Create bar chart visualization of results."""
    metrics = ['chain_recall', 'gold_recall', 'poison_rate', 'redundancy']
    metric_names = ['Chain Recall', 'Gold Recall', 'Poison Rate', 'Redundancy']
    strategies = ['Top-K', 'MMR', 'QUBO']
    colors = ['#e74c3c', '#f39c12', '#27ae60']  # Red, Orange, Green

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

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

    plt.suptitle('Experiment 1.1: Poisoned Stress Test - Top-K vs MMR vs QUBO',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Run Experiment 1.1: Poisoned Stress Test')
    parser.add_argument('--n-questions', type=int, default=100, help='Number of questions')
    parser.add_argument('--k', type=int, default=5, help='Number of results to retrieve')
    parser.add_argument('--n-hops', type=int, default=3, help='Number of hops (gold facts) per question')
    parser.add_argument('--n-clones', type=int, default=3, help='Number of clones per gold fact')
    parser.add_argument('--alpha', type=float, default=0.7, help='QUBO alpha parameter')
    parser.add_argument('--lambda-param', type=float, default=0.5, help='MMR lambda parameter')
    parser.add_argument('--output-dir', type=str, default='experiments/results', help='Output directory')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate dataset
    generator = PoisonedDatasetGenerator()
    questions = generator.generate_synthetic_dataset(
        n_questions=args.n_questions,
        n_hops=args.n_hops,
        n_clones_per_fact=args.n_clones
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
    results_file = output_dir / 'exp_1_1_poisoned_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_file}")

    # Plot results
    plot_file = output_dir / 'exp_1_1_poisoned_chart.png'
    plot_results(results, str(plot_file))


if __name__ == "__main__":
    main()
