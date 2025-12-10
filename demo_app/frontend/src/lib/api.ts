/**
 * API client for Quantum-RAG Demo backend
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface RetrievalResult {
  rank: number;
  score: number;
  text: string;
  source: string;
  chunk_id: string;
}

export interface RetrievalMetrics {
  latency_ms: number;
  intra_list_similarity: number;
  cluster_coverage: number;
  total_clusters: number;
  avg_relevance: number;
}

export interface MethodResult {
  method: string;
  results: RetrievalResult[];
  metrics: RetrievalMetrics;
  llm_response: string | null;
}

export interface UMAPPoint {
  x: number;
  y: number;
  chunk_id: string;
  source: string;
  cluster: string;
  is_selected: boolean;
  selected_by: string[];
}

export interface CompareResponse {
  query: string;
  dataset: string;
  topk: MethodResult;
  mmr: MethodResult;
  qubo: MethodResult;
  umap_points: UMAPPoint[];
  query_point: { x: number; y: number } | null;
}

export interface DatasetInfo {
  name: string;
  total_chunks: number;
  total_clusters: number;
  description: string;
}

export interface HealthResponse {
  status: string;
  orbit_available: boolean;
  datasets_loaded: string[];
}

/**
 * Check if backend is available
 */
export async function checkBackendHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE}/api/health`, {
      method: 'GET',
      signal: AbortSignal.timeout(2000),
    });
    return response.ok;
  } catch {
    return false;
  }
}

/**
 * Compare all three retrieval methods with configurable parameters
 */
export async function compareMethodsAPI(
  query: string,
  dataset: string = 'medical',
  k: number = 5,
  includeLlm: boolean = true,
  alpha: number = 0.02,
  penalty: number = 1000.0,
  lambdaParam: number = 0.5,
  solverPreset: string = 'balanced'
): Promise<CompareResponse> {
  const response = await fetch(`${API_BASE}/api/compare`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      query,
      dataset,
      k,
      include_llm: includeLlm,
      alpha,
      penalty,
      lambda_param: lambdaParam,
      solver_preset: solverPreset,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Comparison failed');
  }

  return response.json();
}

/**
 * Get available datasets
 */
export async function getDatasetsAPI(): Promise<{ datasets: DatasetInfo[] }> {
  const response = await fetch(`${API_BASE}/api/datasets`);

  if (!response.ok) {
    throw new Error('Failed to fetch datasets');
  }

  return response.json();
}

/**
 * Get UMAP embeddings for a dataset
 */
export async function getEmbeddingsAPI(
  dataset: string,
  forceRecompute: boolean = false
): Promise<{ dataset: string; num_documents: number; umap_points: UMAPPoint[]; clusters: string[] }> {
  const url = new URL(`${API_BASE}/api/embeddings/${dataset}`);
  if (forceRecompute) {
    url.searchParams.set('force_recompute', 'true');
  }

  const response = await fetch(url.toString());

  if (!response.ok) {
    throw new Error('Failed to fetch embeddings');
  }

  return response.json();
}

/**
 * Health check
 */
export async function healthCheckAPI(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE}/api/health`);

  if (!response.ok) {
    throw new Error('Health check failed');
  }

  return response.json();
}
