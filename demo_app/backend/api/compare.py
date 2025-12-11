"""
Compare API - Runs all three retrieval methods and returns comparison.
"""

from fastapi import APIRouter, HTTPException

from ..models.schemas import CompareRequest, CompareResponse
from ..services.retrieval_service import get_retrieval_service
from ..services.visualization_service import get_visualization_service

router = APIRouter(prefix="/api", tags=["compare"])


@router.post("/compare", response_model=CompareResponse)
async def compare_methods(request: CompareRequest) -> CompareResponse:
    """
    Run all three retrieval methods (Top-K, MMR, QUBO) and compare results.

    This is the main endpoint for the demo - it shows side-by-side comparison
    of industry standard vs Quantum-RAG.
    """
    try:
        retrieval_service = get_retrieval_service()
        viz_service = get_visualization_service()

        # Check if dataset is available
        available = retrieval_service.get_available_datasets()
        if request.dataset not in available:
            raise HTTPException(
                status_code=400,
                detail=f"Dataset '{request.dataset}' not available. Available: {available}"
            )

        # Run comparison with configurable parameters
        results = await retrieval_service.compare_methods(
            query=request.query,
            dataset=request.dataset,
            k=request.k,
            include_llm=request.include_llm,
            alpha=request.alpha,
            beta=request.beta,
            penalty=request.penalty,
            lambda_param=request.lambda_param,
            solver_preset=request.solver_preset,
        )

        # Get UMAP coordinates
        umap_points = viz_service.get_embeddings_for_dataset(request.dataset)

        # Mark selected points
        selected_ids = {
            "topk": [r.chunk_id for r in results["topk"].results],
            "mmr": [r.chunk_id for r in results["mmr"].results],
            "qubo": [r.chunk_id for r in results["qubo"].results],
        }
        umap_points = viz_service.mark_selected_points(umap_points, selected_ids)

        # Get query point
        query_point = viz_service.get_query_point(request.query, request.dataset)

        return CompareResponse(
            query=request.query,
            dataset=request.dataset,
            topk=results["topk"],
            mmr=results["mmr"],
            qubo=results["qubo"],
            umap_points=umap_points,
            query_point=query_point,
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")
