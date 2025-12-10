"""
Embeddings API - UMAP coordinates and visualization data.
"""

from fastapi import APIRouter, HTTPException

from ..models.schemas import EmbeddingsResponse
from ..services.retrieval_service import get_retrieval_service
from ..services.visualization_service import get_visualization_service

router = APIRouter(prefix="/api", tags=["embeddings"])


@router.get("/embeddings/{dataset}", response_model=EmbeddingsResponse)
async def get_embeddings(dataset: str, force_recompute: bool = False) -> EmbeddingsResponse:
    """
    Get pre-computed UMAP coordinates for all documents in a dataset.

    Used for the embedding space visualization.
    """
    try:
        retrieval_service = get_retrieval_service()
        viz_service = get_visualization_service()

        # Check if dataset is available
        available = retrieval_service.get_available_datasets()
        if dataset not in available:
            raise HTTPException(
                status_code=400,
                detail=f"Dataset '{dataset}' not available. Available: {available}"
            )

        # Get UMAP points
        points = viz_service.get_embeddings_for_dataset(
            dataset,
            force_recompute=force_recompute
        )

        # Get clusters
        clusters = viz_service.get_clusters(dataset)

        return EmbeddingsResponse(
            dataset=dataset,
            num_documents=len(points),
            umap_points=points,
            clusters=clusters,
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get embeddings: {str(e)}")


@router.get("/datasets")
async def list_datasets():
    """
    List available datasets.
    """
    service = get_retrieval_service()
    available = service.get_available_datasets()

    datasets = []
    for name in available:
        stats = service.get_dataset_stats(name)
        config = service.DATASETS.get(name, {})
        datasets.append({
            "name": name,
            "total_chunks": stats.get("total_chunks", 0),
            "total_clusters": config.get("total_clusters", 0),
            "description": config.get("description", ""),
        })

    return {"datasets": datasets}
