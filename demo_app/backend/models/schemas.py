"""
Pydantic models for API request/response schemas.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class RetrievalResult(BaseModel):
    """Single retrieved document result."""
    rank: int = Field(..., description="Rank position (1-indexed)")
    score: float = Field(..., description="Similarity score (0-1)")
    text: str = Field(..., description="Document text content")
    source: str = Field(..., description="Source filename")
    chunk_id: str = Field(..., description="Unique chunk identifier")


class RetrievalMetrics(BaseModel):
    """Metrics for a retrieval method."""
    latency_ms: float = Field(..., description="Execution time in milliseconds")
    intra_list_similarity: float = Field(..., description="Average pairwise similarity (lower=more diverse)")
    cluster_coverage: int = Field(..., description="Number of unique clusters covered")
    total_clusters: int = Field(..., description="Total clusters in dataset")
    avg_relevance: float = Field(..., description="Average relevance score")


class MethodResult(BaseModel):
    """Results from a single retrieval method."""
    method: str = Field(..., description="Method name (topk, mmr, qubo)")
    results: List[RetrievalResult] = Field(..., description="Retrieved documents")
    metrics: RetrievalMetrics = Field(..., description="Performance metrics")
    llm_response: Optional[str] = Field(None, description="LLM-generated answer")


class UMAPPoint(BaseModel):
    """2D UMAP coordinate for visualization."""
    x: float
    y: float
    chunk_id: str
    source: str
    cluster: str
    is_selected: bool = False
    selected_by: List[str] = Field(default_factory=list, description="Methods that selected this")


class CompareRequest(BaseModel):
    """Request for /api/compare endpoint."""
    query: str = Field(..., description="User query", min_length=1)
    dataset: str = Field("medical", description="Dataset to search (medical, legal, greedy_trap)")
    k: int = Field(5, description="Number of results to retrieve", ge=1, le=20)
    include_llm: bool = Field(True, description="Whether to generate LLM responses")


class CompareResponse(BaseModel):
    """Response from /api/compare endpoint."""
    query: str = Field(..., description="Original query")
    dataset: str = Field(..., description="Dataset used")
    topk: MethodResult = Field(..., description="Top-K (naive) results")
    mmr: MethodResult = Field(..., description="MMR results")
    qubo: MethodResult = Field(..., description="QUBO results")
    umap_points: List[UMAPPoint] = Field(..., description="All documents with UMAP coordinates")
    query_point: Optional[Dict[str, float]] = Field(None, description="Query UMAP position")


class RetrieveRequest(BaseModel):
    """Request for /api/retrieve endpoint (single method)."""
    query: str = Field(..., description="User query", min_length=1)
    dataset: str = Field("medical", description="Dataset to search")
    method: str = Field("qubo", description="Retrieval method (topk, mmr, qubo)")
    k: int = Field(5, description="Number of results", ge=1, le=20)


class RetrieveResponse(BaseModel):
    """Response from /api/retrieve endpoint."""
    query: str
    dataset: str
    result: MethodResult


class EmbeddingsResponse(BaseModel):
    """Response from /api/embeddings endpoint."""
    dataset: str
    num_documents: int
    umap_points: List[UMAPPoint]
    clusters: List[str] = Field(..., description="List of unique cluster names")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    orbit_available: bool = True
    datasets_loaded: List[str] = Field(default_factory=list)
