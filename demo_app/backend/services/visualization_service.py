"""
Visualization Service - Handles UMAP computation and caching for embeddings visualization.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from ..models.schemas import UMAPPoint
from .retrieval_service import get_retrieval_service
from core.utils import extract_disease_from_filename


class VisualizationService:
    """
    Service for computing and caching UMAP coordinates for visualization.
    """

    def __init__(self, cache_dir: str = "./umap_cache"):
        """
        Initialize visualization service.

        Args:
            cache_dir: Directory for caching UMAP coordinates
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, dataset: str) -> Path:
        """Get cache file path for dataset."""
        return self.cache_dir / f"{dataset}_umap.json"

    def _load_cache(self, dataset: str) -> Optional[Dict[str, Any]]:
        """Load cached UMAP data if available."""
        cache_path = self._get_cache_path(dataset)
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def _save_cache(self, dataset: str, data: Dict[str, Any]):
        """Save UMAP data to cache."""
        cache_path = self._get_cache_path(dataset)
        with open(cache_path, 'w') as f:
            json.dump(data, f)

    def compute_umap(
        self,
        embeddings: np.ndarray,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "cosine",
    ) -> np.ndarray:
        """
        Compute UMAP 2D projection of embeddings.

        Args:
            embeddings: Array of embeddings (n_samples, n_features)
            n_neighbors: UMAP n_neighbors parameter
            min_dist: UMAP min_dist parameter
            metric: Distance metric

        Returns:
            2D coordinates (n_samples, 2)
        """
        try:
            import umap
        except ImportError:
            # Fallback to PCA if UMAP not available
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            return pca.fit_transform(embeddings)

        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            n_components=2,
            random_state=42,
        )
        return reducer.fit_transform(embeddings)

    def get_embeddings_for_dataset(
        self,
        dataset: str,
        force_recompute: bool = False,
    ) -> List[UMAPPoint]:
        """
        Get UMAP coordinates for all documents in a dataset.

        Args:
            dataset: Dataset name
            force_recompute: Force recomputation even if cached

        Returns:
            List of UMAPPoint objects
        """
        # Check cache first
        if not force_recompute:
            cached = self._load_cache(dataset)
            if cached:
                return [UMAPPoint(**p) for p in cached["points"]]

        # Get retrieval service and ensure dataset is indexed
        service = get_retrieval_service()
        service.index_dataset(dataset)

        # Get vector store
        vector_store = service.get_vector_store(dataset)

        # Get all embeddings and metadata
        embeddings, metadata = vector_store.get_all_embeddings()

        if len(embeddings) == 0:
            return []

        # Compute UMAP
        coords = self.compute_umap(embeddings)

        # Build UMAPPoint objects
        points = []
        for i, meta in enumerate(metadata):
            source = meta.get("source", "unknown")
            cluster = extract_disease_from_filename(source)

            points.append(UMAPPoint(
                x=float(coords[i, 0]),
                y=float(coords[i, 1]),
                chunk_id=meta["id"],
                source=source,
                cluster=cluster,
                is_selected=False,
                selected_by=[],
            ))

        # Cache results
        self._save_cache(dataset, {
            "points": [p.model_dump() for p in points],
        })

        return points

    def get_query_point(
        self,
        query: str,
        dataset: str,
    ) -> Optional[Dict[str, float]]:
        """
        Get UMAP coordinates for a query by projecting it into existing space.

        This is an approximation - we find the nearest neighbor and use its coordinates.
        For a more accurate projection, we would need to re-run UMAP with the query included.

        Args:
            query: Query text
            dataset: Dataset name

        Returns:
            Dict with x, y coordinates or None
        """
        service = get_retrieval_service()
        service.index_dataset(dataset)

        # Embed query
        query_embedding = service.embedder.embed_query(query)

        # Get nearest neighbor
        vector_store = service.get_vector_store(dataset)
        nearest = vector_store.search(query_embedding, k=1)

        if not nearest:
            return None

        # Get UMAP points
        points = self.get_embeddings_for_dataset(dataset)

        # Find the nearest point
        nearest_id = nearest[0]["id"]
        for point in points:
            if point.chunk_id == nearest_id:
                # Return slightly offset from nearest neighbor
                return {
                    "x": point.x + 0.1,
                    "y": point.y + 0.1,
                }

        return None

    def mark_selected_points(
        self,
        points: List[UMAPPoint],
        selected_ids: Dict[str, List[str]],
    ) -> List[UMAPPoint]:
        """
        Mark which points were selected by each method.

        Args:
            points: List of UMAP points
            selected_ids: Dict mapping method name to list of selected chunk IDs

        Returns:
            Updated list of UMAP points with selection info
        """
        # Build lookup of selected IDs per method
        id_to_methods = {}
        for method, ids in selected_ids.items():
            for chunk_id in ids:
                if chunk_id not in id_to_methods:
                    id_to_methods[chunk_id] = []
                id_to_methods[chunk_id].append(method)

        # Update points
        updated_points = []
        for point in points:
            methods = id_to_methods.get(point.chunk_id, [])
            updated_points.append(UMAPPoint(
                x=point.x,
                y=point.y,
                chunk_id=point.chunk_id,
                source=point.source,
                cluster=point.cluster,
                is_selected=len(methods) > 0,
                selected_by=methods,
            ))

        return updated_points

    def get_clusters(self, dataset: str) -> List[str]:
        """
        Get list of unique clusters in a dataset.

        Args:
            dataset: Dataset name

        Returns:
            List of cluster names
        """
        points = self.get_embeddings_for_dataset(dataset)
        clusters = list(set(p.cluster for p in points))
        return sorted(clusters)


# Global service instance
_viz_service_instance: Optional[VisualizationService] = None


def get_visualization_service() -> VisualizationService:
    """Get or create the global visualization service instance."""
    global _viz_service_instance
    if _viz_service_instance is None:
        _viz_service_instance = VisualizationService()
    return _viz_service_instance
