"""
Quantum-RAG Demo API

FastAPI backend for the investor demo application.
Provides endpoints for comparing Top-K, MMR, and QUBO retrieval methods.
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demo_app.backend.api import compare_router, retrieve_router, embeddings_router
from demo_app.backend.models.schemas import HealthResponse
from demo_app.backend.services.retrieval_service import get_retrieval_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Pre-loads models and indexes datasets on startup.
    """
    print("Starting Quantum-RAG Demo API...")

    # Pre-load retrieval service (loads embedding model)
    service = get_retrieval_service()

    # Pre-index available datasets
    available = service.get_available_datasets()
    print(f"Available datasets: {available}")

    for dataset in available:
        try:
            count = service.index_dataset(dataset)
            print(f"  - {dataset}: {count} documents indexed")
        except Exception as e:
            print(f"  - {dataset}: Failed to index ({e})")

    print("API ready!")
    yield
    print("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Quantum-RAG Demo API",
    description="""
    Backend API for the Quantum-RAG investor demo application.

    ## Features

    - **Compare Methods**: Run Top-K, MMR, and QUBO retrieval side-by-side
    - **UMAP Visualization**: Get 2D embeddings for visualization
    - **Real-time Metrics**: Latency, diversity, cluster coverage

    ## Endpoints

    - `POST /api/compare` - Compare all three methods
    - `POST /api/retrieve` - Run single method
    - `GET /api/embeddings/{dataset}` - Get UMAP coordinates
    - `GET /api/datasets` - List available datasets
    - `GET /api/health` - Health check
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js dev server
        "http://127.0.0.1:3000",
        "http://localhost:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(compare_router)
app.include_router(retrieve_router)
app.include_router(embeddings_router)


@app.get("/api/health", response_model=HealthResponse, tags=["health"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns status of the API and available datasets.
    """
    service = get_retrieval_service()
    available = service.get_available_datasets()

    # Check ORBIT availability
    orbit_available = True
    try:
        import orbit
    except ImportError:
        orbit_available = False

    return HealthResponse(
        status="healthy",
        orbit_available=orbit_available,
        datasets_loaded=available,
    )


@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Quantum-RAG Demo API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
