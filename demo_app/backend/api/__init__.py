from .compare import router as compare_router
from .retrieve import router as retrieve_router
from .embeddings import router as embeddings_router

__all__ = ["compare_router", "retrieve_router", "embeddings_router"]
