"""
Experiment 1.3: The Dose-Response Curve

Purpose: Quantify how retrieval quality degrades as redundancy increases.
         Shows the "breaking point" for each method.

Hypothesis:
- Top-K: Steep decline in chain recall as clones increase
- MMR: Moderate decline, hits floor around 2-3 clones
- QUBO: Flat curve, maintains high recall regardless of clone count

Dataset: 5 variants of poisoned MuSiQue
- 0 clones: Clean baseline
- 1 clone per gold fact: Mild redundancy
- 2 clones per gold fact: Moderate redundancy
- 3 clones per gold fact: Heavy redundancy
- 5 clones per gold fact: Extreme redundancy

Comparison: Top-K vs MMR vs QUBO

Metrics:
- Chain Recall at each redundancy level
- Gold Recall at each redundancy level
- Poison Rate at each redundancy level

Output: Line chart showing decline curves for each method

Usage:
    python scripts/run_dose_response_experiment.py
    python scripts/run_dose_response_experiment.py --n-questions 50
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
class PoisonClone:
    """A semantic clone of a gold fact."""
    id: str
    text: str
    source_gold_id: str
    embedding: np.ndarray = None


@dataclass
class Question:
    """A multi-hop question with gold facts and optional poison clones."""
    id: str
    query: str
    gold_facts: List[GoldFact]
    poison_clones: List[PoisonClone]
    distractor_paragraphs: List[Dict]


class DoseResponseDatasetGenerator:
    """
    Generates datasets with varying levels of redundancy (clone counts).
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        print("Loading embedding model...")
        self.model = SentenceTransformer(model_name)

    def generate_dataset_with_clones(
        self,
        n_questions: int = 100,
        n_hops: int = 3,
        n_clones_per_fact: int = 0,
        n_distractors: int = 10
    ) -> List[Question]:
        """Generate dataset with specified number of clones per gold fact."""

        topics = ["history", "science", "geography", "culture", "technology",
                  "medicine", "economics", "literature", "sports", "music"]

        questions = []

        for q_idx in range(n_questions):
            topic_name = topics[q_idx % len(topics)]

            # Generate gold facts
            gold_facts = []
            for hop in range(n_hops):
                fact_text = f"[GOLD-{q_idx}-{hop}] Gold fact {hop+1} for question {q_idx} " \
                           f"about {topic_name}. Hop {hop+1} of {n_hops}. " \
                           f"Essential info: GOLD_HOP{hop}_Q{q_idx}."

                gold_facts.append(GoldFact(
                    id=f"gold_{q_idx}_{hop}",
                    text=fact_text
                ))

            # Generate poison clones (if any)
            poison_clones = []
            if n_clones_per_fact > 0:
                for gold_fact in gold_facts:
                    for clone_idx in range(n_clones_per_fact):
                        clone_text = self._paraphrase(gold_fact.text, clone_idx)
                        poison_clones.append(PoisonClone(
                            id=f"clone_{gold_fact.id}_{clone_idx}",
                            text=clone_text,
                            source_gold_id=gold_fact.id
                        ))

            # Generate distractors
            distractors = []
            distractor_topics = ["astronomy", "cooking", "politics", "fashion",
                                "architecture", "biology", "chemistry", "physics"]

            for d_idx in range(n_distractors):
                d_topic = distractor_topics[d_idx % len(distractor_topics)]
                distractor_text = f"[DISTRACTOR-{q_idx}-{d_idx}] About {d_topic}, " \
                                 f"unrelated to {topic_name}. Content {d_idx}."

                distractors.append({
                    'id': f"distractor_{q_idx}_{d_idx}",
                    'text': distractor_text,
                    'is_gold': False,
                    'is_clone': False
                })

            query = f"Multi-hop question {q_idx}: What is the chain of facts " \
                   f"about {topic_name} with {n_hops} reasoning steps?"

            questions.append(Question(
                id=f"q_{q_idx}",
                query=query,
                gold_facts=gold_facts,
                poison_clones=poison_clones,
                distractor_paragraphs=distractors
            ))

        # Compute embeddings
        self._compute_embeddings(questions)

        return questions

    def _paraphrase(self, text: str, variant: int) -> str:
        """Create a paraphrase of the text."""
        paraphrase_patterns = [
            lambda t: t.replace("Gold fact", "Important information").replace("Essential", "Key"),
            lambda t: t.replace("Gold fact", "Critical data").replace("Essential", "Vital"),
            lambda t: t.replace("Gold fact", "Crucial detail").replace("Essential", "Important"),
            lambda t: t.replace("Gold fact", "Key finding").replace("Essential", "Core"),
            lambda t: t.replace("Gold fact", "Main point").replace("Essential", "Primary"),
        ]

        pattern = paraphrase_patterns[variant % len(paraphrase_patterns)]
        paraphrased = pattern(text)
        paraphrased = paraphrased.replace("[GOLD-", f"[CLONE-v{variant}-")

        return paraphrased

    def _compute_embeddings(self, questions: List[Question]):
        """Compute embeddings for all texts."""
        all_texts = []
        text_mapping = []

        for q_idx, question in enumerate(questions):
            all_texts.append(question.query)
            text_mapping.append((q_idx, 'query', 0))

            for g_idx, gold in enumerate(question.gold_facts):
                all_texts.append(gold.text)
                text_mapping.append((q_idx, 'gold', g_idx))

            for c_idx, clone in enumerate(question.poison_clones):
                all_texts.append(clone.text)
                text_mapping.append((q_idx, 'clone', c_idx))

            for d_idx, distractor in enumerate(question.distractor_paragraphs):
                all_texts.append(distractor['text'])
                text_mapping.append((q_idx, 'distractor', d_idx))

        embeddings = self.model.encode(all_texts, show_progress_bar=True, convert_to_numpy=True)

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

    query_norm = query_embedding / np.linalg.norm(query_embedding)
    candidate_norms = candidate_embeddings / np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)

    similarities = np.dot(candidate_norms, query_norm)

    for candidate, sim in zip(candidates, similarities):
        candidate['score'] = float(sim)

    candidates.sort(key=lambda x: x['score'], reverse=True)

    return candidates


def evaluate_retrieval(retrieved_ids: Set[str], question: Question) -> Dict[str, float]:
    """Evaluate retrieval results."""
    gold_ids = {g.id for g in question.gold_facts}
    clone_ids = {c.id for c in question.poison_clones}

    gold_retrieved = retrieved_ids & gold_ids
    gold_recall = len(gold_retrieved) / len(gold_ids) if gold_ids else 0
    chain_recall = 1.0 if gold_retrieved == gold_ids else 0.0

    clones_retrieved = retrieved_ids & clone_ids
    poison_rate = len(clones_retrieved) / len(retrieved_ids) if retrieved_ids else 0

    return {
        'chain_recall': chain_recall,
        'gold_recall': gold_recall,
        'poison_rate': poison_rate
    }


def run_single_experiment(
    questions: List[Question],
    strategies: Dict,
    k: int = 5
) -> Dict[str, Dict[str, float]]:
    """Run experiment on a single dataset variant."""
    results = {name: defaultdict(list) for name in strategies.keys()}

    for question in questions:
        candidates = build_candidate_pool(question)
        candidates = compute_query_similarities(question.query_embedding, candidates)

        for strategy_name, strategy in strategies.items():
            retrieved_results, _ = strategy.retrieve(
                query_embedding=question.query_embedding,
                candidate_results=candidates,
                k=k
            )

            retrieved_ids = {r.id for r in retrieved_results}
            eval_metrics = evaluate_retrieval(retrieved_ids, question)

            results[strategy_name]['chain_recall'].append(eval_metrics['chain_recall'])
            results[strategy_name]['gold_recall'].append(eval_metrics['gold_recall'])
            results[strategy_name]['poison_rate'].append(eval_metrics['poison_rate'])

    # Aggregate
    aggregated = {}
    for strategy_name, metrics in results.items():
        aggregated[strategy_name] = {
            'chain_recall': np.mean(metrics['chain_recall']),
            'chain_recall_std': np.std(metrics['chain_recall']),
            'gold_recall': np.mean(metrics['gold_recall']),
            'gold_recall_std': np.std(metrics['gold_recall']),
            'poison_rate': np.mean(metrics['poison_rate']),
            'poison_rate_std': np.std(metrics['poison_rate'])
        }

    return aggregated


def run_dose_response_experiment(
    n_questions: int = 100,
    n_hops: int = 3,
    k: int = 5,
    alpha: float = 0.7,
    lambda_param: float = 0.5
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """
    Run the dose-response experiment across all clone levels.
    """
    clone_levels = [0, 1, 2, 3, 5]

    # Create strategies
    strategies = {
        'Top-K': create_retrieval_strategy('naive'),
        'MMR': create_retrieval_strategy('mmr', lambda_param=lambda_param),
        'QUBO': create_retrieval_strategy('qubo', alpha=alpha, solver_preset='balanced')
    }

    generator = DoseResponseDatasetGenerator()
    all_results = {}

    for n_clones in clone_levels:
        print(f"\n{'='*60}")
        print(f"Running with {n_clones} clones per gold fact...")
        print(f"{'='*60}")

        questions = generator.generate_dataset_with_clones(
            n_questions=n_questions,
            n_hops=n_hops,
            n_clones_per_fact=n_clones
        )

        results = run_single_experiment(questions, strategies, k)
        all_results[n_clones] = results

        # Print intermediate results
        print(f"\nResults for {n_clones} clones:")
        for strategy in ['Top-K', 'MMR', 'QUBO']:
            cr = results[strategy]['chain_recall']
            print(f"  {strategy}: Chain Recall = {cr:.3f}")

    return all_results


def print_results(all_results: Dict[int, Dict[str, Dict[str, float]]]):
    """Print experiment results as a table."""
    print("\n" + "="*80)
    print("EXPERIMENT 1.3: DOSE-RESPONSE CURVE RESULTS")
    print("="*80)

    clone_levels = sorted(all_results.keys())
    strategies = ['Top-K', 'MMR', 'QUBO']

    # Chain Recall table
    print("\n--- Chain Recall by Clone Count ---")
    header = f"{'Clones':<10}"
    for s in strategies:
        header += f"{s:<15}"
    print(header)
    print("-" * 55)

    for n_clones in clone_levels:
        row = f"{n_clones:<10}"
        for strategy in strategies:
            cr = all_results[n_clones][strategy]['chain_recall']
            std = all_results[n_clones][strategy]['chain_recall_std']
            row += f"{cr:.3f} ± {std:.2f}   "
        print(row)

    # Gold Recall table
    print("\n--- Gold Recall by Clone Count ---")
    header = f"{'Clones':<10}"
    for s in strategies:
        header += f"{s:<15}"
    print(header)
    print("-" * 55)

    for n_clones in clone_levels:
        row = f"{n_clones:<10}"
        for strategy in strategies:
            gr = all_results[n_clones][strategy]['gold_recall']
            std = all_results[n_clones][strategy]['gold_recall_std']
            row += f"{gr:.3f} ± {std:.2f}   "
        print(row)

    # Poison Rate table
    print("\n--- Poison Rate by Clone Count ---")
    header = f"{'Clones':<10}"
    for s in strategies:
        header += f"{s:<15}"
    print(header)
    print("-" * 55)

    for n_clones in clone_levels:
        row = f"{n_clones:<10}"
        for strategy in strategies:
            pr = all_results[n_clones][strategy]['poison_rate']
            std = all_results[n_clones][strategy]['poison_rate_std']
            row += f"{pr:.3f} ± {std:.2f}   "
        print(row)

    print("="*80)

    # Analysis
    print("\nAnalysis:")
    topk_decline = all_results[0]['Top-K']['chain_recall'] - all_results[5]['Top-K']['chain_recall']
    mmr_decline = all_results[0]['MMR']['chain_recall'] - all_results[5]['MMR']['chain_recall']
    qubo_decline = all_results[0]['QUBO']['chain_recall'] - all_results[5]['QUBO']['chain_recall']

    print(f"  Top-K decline (0→5 clones): {topk_decline:.3f}")
    print(f"  MMR decline (0→5 clones): {mmr_decline:.3f}")
    print(f"  QUBO decline (0→5 clones): {qubo_decline:.3f}")

    if qubo_decline < topk_decline / 2:
        print("  ✓ QUBO shows significantly less degradation than Top-K")
    if qubo_decline < mmr_decline / 2:
        print("  ✓ QUBO shows significantly less degradation than MMR")

    print("="*80)


def plot_results(all_results: Dict[int, Dict[str, Dict[str, float]]], output_file: str):
    """Create line chart visualization of dose-response curves."""
    clone_levels = sorted(all_results.keys())
    strategies = ['Top-K', 'MMR', 'QUBO']
    colors = {'Top-K': '#e74c3c', 'MMR': '#f39c12', 'QUBO': '#27ae60'}
    markers = {'Top-K': 'o', 'MMR': 's', 'QUBO': '^'}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = [
        ('chain_recall', 'Chain Recall', 'Higher is better'),
        ('gold_recall', 'Gold Recall', 'Higher is better'),
        ('poison_rate', 'Poison Rate', 'Lower is better')
    ]

    for ax, (metric, title, note) in zip(axes, metrics):
        for strategy in strategies:
            values = [all_results[n][strategy][metric] for n in clone_levels]
            errors = [all_results[n][strategy][f'{metric}_std'] for n in clone_levels]

            ax.errorbar(
                clone_levels, values,
                yerr=errors,
                label=strategy,
                color=colors[strategy],
                marker=markers[strategy],
                markersize=8,
                linewidth=2,
                capsize=4
            )

        ax.set_xlabel('Clones per Gold Fact')
        ax.set_ylabel(title)
        ax.set_title(f'{title}\n({note})')
        ax.set_xticks(clone_levels)
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Experiment 1.3: Dose-Response Curve - How Redundancy Affects Retrieval',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Run Experiment 1.3: Dose-Response Curve')
    parser.add_argument('--n-questions', type=int, default=100, help='Number of questions per variant')
    parser.add_argument('--k', type=int, default=5, help='Number of results to retrieve')
    parser.add_argument('--n-hops', type=int, default=3, help='Number of hops (gold facts) per question')
    parser.add_argument('--alpha', type=float, default=0.7, help='QUBO alpha parameter')
    parser.add_argument('--lambda-param', type=float, default=0.5, help='MMR lambda parameter')
    parser.add_argument('--output-dir', type=str, default='experiments/results', help='Output directory')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run dose-response experiment
    all_results = run_dose_response_experiment(
        n_questions=args.n_questions,
        n_hops=args.n_hops,
        k=args.k,
        alpha=args.alpha,
        lambda_param=args.lambda_param
    )

    # Print results
    print_results(all_results)

    # Save results (convert int keys to strings for JSON)
    results_for_json = {str(k): v for k, v in all_results.items()}
    results_file = output_dir / 'exp_1_3_dose_response_results.json'
    with open(results_file, 'w') as f:
        json.dump(results_for_json, f, indent=2)
    print(f"\nSaved results to {results_file}")

    # Plot results
    plot_file = output_dir / 'exp_1_3_dose_response_chart.png'
    plot_results(all_results, str(plot_file))


if __name__ == "__main__":
    main()
