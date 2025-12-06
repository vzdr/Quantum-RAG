"""
Validate Similarity Structure of Strategic Synthetic Data

This script validates that generated data achieves target similarity ranges:
- Intra-cluster similarity: 0.65-0.80 (moderate, not too high)
- Inter-cluster similarity: 0.30-0.50 (diverse)
- Multiple high-relevance clusters for query

Visualizes similarity matrices and reports deviations from targets.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Warning: sentence-transformers not installed. Install with: pip install sentence-transformers")
    SentenceTransformer = None


class SimilarityValidator:
    """Validates similarity structure of generated data."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize with embedding model."""
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers required. Install with: pip install sentence-transformers")

        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def load_documents(self, directory: Path) -> Dict[str, str]:
        """Load all text documents from directory."""
        documents = {}
        for filepath in directory.glob("*.txt"):
            with open(filepath, 'r', encoding='utf-8') as f:
                documents[filepath.name] = f.read()
        return documents

    def compute_embeddings(self, documents: Dict[str, str]) -> Tuple[List[str], np.ndarray]:
        """Compute embeddings for all documents."""
        filenames = list(documents.keys())
        texts = [documents[fn] for fn in filenames]

        print(f"Computing embeddings for {len(texts)} documents...")
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

        return filenames, embeddings

    def compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute pairwise cosine similarity matrix."""
        # Normalize embeddings
        norms = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Compute similarities
        similarity_matrix = np.dot(norms, norms.T)
        return similarity_matrix

    def identify_clusters(self, filenames: List[str], metadata: Dict) -> Dict[str, List[int]]:
        """Map cluster names to document indices."""
        clusters = metadata.get('clusters', {})
        cluster_indices = {}

        for cluster_name, cluster_info in clusters.items():
            cluster_docs = cluster_info.get('docs', [])
            indices = []
            for i, fn in enumerate(filenames):
                if fn in cluster_docs:
                    indices.append(i)
            cluster_indices[cluster_name] = indices

        return cluster_indices

    def compute_intra_cluster_similarity(self, sim_matrix: np.ndarray, cluster_indices: List[int]) -> float:
        """Compute average intra-cluster similarity."""
        if len(cluster_indices) < 2:
            return 0.0

        similarities = []
        for i in range(len(cluster_indices)):
            for j in range(i + 1, len(cluster_indices)):
                idx_i = cluster_indices[i]
                idx_j = cluster_indices[j]
                similarities.append(sim_matrix[idx_i, idx_j])

        return float(np.mean(similarities))

    def compute_inter_cluster_similarity(self, sim_matrix: np.ndarray,
                                        cluster1_indices: List[int],
                                        cluster2_indices: List[int]) -> float:
        """Compute average similarity between two clusters."""
        similarities = []
        for i in cluster1_indices:
            for j in cluster2_indices:
                similarities.append(sim_matrix[i, j])

        return float(np.mean(similarities))

    def validate_scenario(self, scenario_dir: Path, metadata: Dict) -> Dict:
        """Validate similarity structure for a scenario."""
        print(f"\n{'='*60}")
        print(f"Validating: {scenario_dir.name}")
        print(f"{'='*60}")

        # Load documents
        documents = self.load_documents(scenario_dir)
        if not documents:
            print(f"  ✗ No documents found in {scenario_dir}")
            return {}

        print(f"  Loaded {len(documents)} documents")

        # Compute embeddings
        filenames, embeddings = self.compute_embeddings(documents)

        # Compute similarity matrix
        sim_matrix = self.compute_similarity_matrix(embeddings)

        # Identify clusters
        cluster_indices = self.identify_clusters(filenames, metadata)
        print(f"  Identified {len(cluster_indices)} clusters")

        # Validate intra-cluster similarities
        print(f"\n  Intra-Cluster Similarities (Target: 0.65-0.80):")
        intra_cluster_sims = {}
        for cluster_name, indices in cluster_indices.items():
            if len(indices) < 2:
                print(f"    {cluster_name}: Only 1 document, skipping")
                continue

            intra_sim = self.compute_intra_cluster_similarity(sim_matrix, indices)
            intra_cluster_sims[cluster_name] = intra_sim

            # Check if within target range
            status = "✓" if 0.65 <= intra_sim <= 0.80 else "✗"
            print(f"    {status} {cluster_name}: {intra_sim:.3f}")

        # Validate inter-cluster similarities
        print(f"\n  Inter-Cluster Similarities (Target: 0.30-0.50):")
        cluster_names = list(cluster_indices.keys())
        inter_cluster_sims = {}

        for i in range(len(cluster_names)):
            for j in range(i + 1, len(cluster_names)):
                name1, name2 = cluster_names[i], cluster_names[j]
                inter_sim = self.compute_inter_cluster_similarity(
                    sim_matrix,
                    cluster_indices[name1],
                    cluster_indices[name2]
                )
                inter_cluster_sims[f"{name1} <-> {name2}"] = inter_sim

                status = "✓" if 0.30 <= inter_sim <= 0.50 else "✗"
                print(f"    {status} {name1} <-> {name2}: {inter_sim:.3f}")

        # Validate query similarities (if query provided)
        query = metadata.get('query')
        query_sims = {}
        if query:
            print(f"\n  Query Similarities:")
            print(f"    Query: \"{query}\"")
            query_emb = self.model.encode([query], convert_to_numpy=True)[0]
            query_emb_norm = query_emb / np.linalg.norm(query_emb)

            for cluster_name, indices in cluster_indices.items():
                cluster_embs = embeddings[indices]
                cluster_embs_norm = cluster_embs / np.linalg.norm(cluster_embs, axis=1, keepdims=True)
                cluster_query_sims = np.dot(cluster_embs_norm, query_emb_norm)
                avg_query_sim = float(np.mean(cluster_query_sims))

                query_sims[cluster_name] = avg_query_sim
                print(f"      {cluster_name}: {avg_query_sim:.3f}")

        # Visualize similarity matrix
        self.plot_similarity_matrix(sim_matrix, filenames, cluster_indices, scenario_dir.name)

        return {
            'scenario': scenario_dir.name,
            'n_documents': len(documents),
            'n_clusters': len(cluster_indices),
            'intra_cluster_similarities': intra_cluster_sims,
            'inter_cluster_similarities': inter_cluster_sims,
            'query_similarities': query_sims,
            'similarity_matrix': sim_matrix.tolist(),
            'filenames': filenames
        }

    def plot_similarity_matrix(self, sim_matrix: np.ndarray, filenames: List[str],
                              cluster_indices: Dict[str, List[int]], scenario_name: str):
        """Plot similarity matrix with cluster boundaries."""
        plt.figure(figsize=(12, 10))

        # Reorder by clusters for visualization
        ordered_indices = []
        cluster_labels = []
        cluster_colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_indices)))

        for cluster_name in cluster_indices.keys():
            indices = cluster_indices[cluster_name]
            ordered_indices.extend(indices)
            cluster_labels.extend([cluster_name] * len(indices))

        # Reorder similarity matrix
        ordered_sim = sim_matrix[np.ix_(ordered_indices, ordered_indices)]

        # Plot heatmap
        sns.heatmap(ordered_sim, cmap='RdYlBu_r', vmin=0, vmax=1,
                   cbar_kws={'label': 'Cosine Similarity'},
                   square=True)

        # Add cluster boundaries
        cluster_boundaries = []
        current_pos = 0
        for cluster_name in cluster_indices.keys():
            n_docs = len(cluster_indices[cluster_name])
            current_pos += n_docs
            cluster_boundaries.append(current_pos)

            # Draw boundary lines
            plt.axhline(current_pos, color='black', linewidth=2)
            plt.axvline(current_pos, color='black', linewidth=2)

        # Add cluster labels
        current_pos = 0
        for i, cluster_name in enumerate(cluster_indices.keys()):
            n_docs = len(cluster_indices[cluster_name])
            mid_pos = current_pos + n_docs / 2
            plt.text(-1, mid_pos, cluster_name, rotation=0, va='center', fontweight='bold')
            current_pos += n_docs

        plt.title(f"Similarity Matrix: {scenario_name}", fontsize=14, fontweight='bold')
        plt.xlabel("Document Index")
        plt.ylabel("Document Index")
        plt.tight_layout()

        # Save figure
        output_path = Path(f"validation_{scenario_name.replace(' ', '_')}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n  ✓ Saved similarity matrix visualization to {output_path}")
        plt.close()


def main():
    """Main validation routine."""
    print("\n" + "="*60)
    print("Strategic Data Similarity Validator")
    print("="*60)

    # Load metadata
    base_dir = Path("data/medical_diagnosis/strategic")
    metadata_file = base_dir / "scenarios_metadata.json"

    if not metadata_file.exists():
        print(f"\n✗ Metadata file not found: {metadata_file}")
        print("  Run generate_strategic_data.py first")
        return

    with open(metadata_file, 'r') as f:
        all_metadata = json.load(f)

    # Initialize validator
    validator = SimilarityValidator()

    # Validate each scenario
    validation_results = {}

    for scenario_key, metadata in all_metadata.items():
        scenario_name = metadata['scenario_name']

        if scenario_name == 'symptom_overlap_trap':
            scenario_dir = base_dir / "scenario_a_symptom_overlap"
        elif scenario_name == 'diagnostic_confusion_trap':
            scenario_dir = base_dir / "scenario_b_diagnostic_confusion"
        else:
            print(f"\nSkipping unknown scenario: {scenario_name}")
            continue

        if not scenario_dir.exists():
            print(f"\n✗ Scenario directory not found: {scenario_dir}")
            continue

        results = validator.validate_scenario(scenario_dir, metadata)
        validation_results[scenario_key] = results

    # Summary report
    print("\n" + "="*60)
    print("Validation Summary")
    print("="*60)

    for scenario_key, results in validation_results.items():
        if not results:
            continue

        print(f"\n{results['scenario']}:")
        print(f"  Documents: {results['n_documents']}")
        print(f"  Clusters: {results['n_clusters']}")

        # Check if intra-cluster sims are in range
        intra_sims = results['intra_cluster_similarities']
        intra_in_range = sum(1 for v in intra_sims.values() if 0.65 <= v <= 0.80)
        print(f"  Intra-cluster in range: {intra_in_range}/{len(intra_sims)}")

        # Check if inter-cluster sims are in range
        inter_sims = results['inter_cluster_similarities']
        inter_in_range = sum(1 for v in inter_sims.values() if 0.30 <= v <= 0.50)
        print(f"  Inter-cluster in range: {inter_in_range}/{len(inter_sims)}")

    # Save validation results
    output_file = base_dir / "validation_results.json"
    with open(output_file, 'w') as f:
        json.dump(validation_results, f, indent=2)
    print(f"\n✓ Saved validation results to {output_file}")

    print("\n" + "="*60)
    print("Validation Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
