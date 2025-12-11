"""
Retrieve API - Single method retrieval endpoint.
"""

from fastapi import APIRouter, HTTPException

from ..models.schemas import RetrieveRequest, RetrieveResponse
from ..services.retrieval_service import get_retrieval_service

router = APIRouter(prefix="/api", tags=["retrieve"])


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_single(request: RetrieveRequest) -> RetrieveResponse:
    """
    Run a single retrieval method.

    Useful for testing individual methods or when you don't need full comparison.
    """
    try:
        service = get_retrieval_service()

        # Validate method
        valid_methods = ["topk", "mmr", "qubo"]
        if request.method not in valid_methods:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid method '{request.method}'. Valid: {valid_methods}"
            )

        # Check if dataset is available
        available = service.get_available_datasets()
        if request.dataset not in available:
            raise HTTPException(
                status_code=400,
                detail=f"Dataset '{request.dataset}' not available. Available: {available}"
            )

        # Run retrieval
        result = await service.retrieve_single(
            query=request.query,
            dataset=request.dataset,
            method=request.method,
            k=request.k,
        )

        return RetrieveResponse(
            query=request.query,
            dataset=request.dataset,
            result=result,
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")
