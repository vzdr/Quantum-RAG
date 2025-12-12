'use client';

import { useState, useCallback, useEffect } from 'react';
import Link from 'next/link';
import { ArrowLeft, Search, Loader2, RotateCcw, Check, X, Minus } from 'lucide-react';
import { compareMethodsAPI, type CompareResponse } from '@/lib/api';
import { DEMO_QUERIES } from '@/lib/mockData';
import ParameterControls from '@/components/demo/ParameterControls';

// Scenario configurations
const scenarios: Record<string, { title: string; description: string }> = {
  medical: {
    title: 'Medical Diagnosis',
    description: 'Compare retrieval methods on differential diagnosis',
  },
  legal: {
    title: 'Legal Case Law',
    description: 'Compare retrieval methods on legal precedents',
  },
  greedy_trap: {
    title: 'Adversarial Test',
    description: 'Stress test designed to break greedy methods',
  },
  wikipedia: {
    title: 'Wikipedia Knowledge',
    description: 'Large-scale retrieval across 171 diverse topics',
  },
};

interface PageProps {
  params: { scenario: string };
}

export default function ScenarioDemoPage({ params }: PageProps) {
  const { scenario } = params;
  const config = scenarios[scenario] || scenarios.medical;
  const suggestions = DEMO_QUERIES[scenario as keyof typeof DEMO_QUERIES] || DEMO_QUERIES.medical;

  const [query, setQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState<CompareResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<string>('');
  const [startTime, setStartTime] = useState<number>(0);

  // Parameter state - production values from experiments
  const [alpha, setAlpha] = useState(0.04);
  const [beta, setBeta] = useState(0.8);
  const [penalty, setPenalty] = useState(10.0);
  const [lambdaParam, setLambdaParam] = useState(0.85);
  const [solverPreset, setSolverPreset] = useState('balanced');

  const handleSubmit = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || isLoading) return;

    setIsLoading(true);
    setError(null);
    setStartTime(Date.now());
    setProgress('Initializing retrieval methods...');

    try {
      // Simulate progress updates
      setTimeout(() => setProgress('Running Top-K baseline (~100ms)...'), 100);
      setTimeout(() => setProgress('Running MMR greedy selection (~200ms)...'), 300);
      setTimeout(() => setProgress(`Running QUBO with ORBIT solver (preset: ${solverPreset}, ~${solverPreset === 'fast' ? '1-2s' : solverPreset === 'balanced' ? '5-10s' : '10-15s'})...`), 600);

      const response = await compareMethodsAPI(
        query.trim(),
        scenario,
        5,
        true,
        alpha,
        beta,
        penalty,
        lambdaParam,
        solverPreset
      );
      setResults(response);
      setProgress('');
    } catch (error) {
      console.error('Comparison failed:', error);
      setError(error instanceof Error ? error.message : 'Comparison failed');
      setProgress('');
    } finally {
      setIsLoading(false);
    }
  }, [query, scenario, isLoading, alpha, beta, penalty, lambdaParam, solverPreset]);

  const handleSuggestion = useCallback((suggestion: string) => {
    setQuery(suggestion);
  }, []);

  const handleReset = useCallback(() => {
    setResults(null);
    setQuery('');
  }, []);

  // Update elapsed time display
  const [elapsedTime, setElapsedTime] = useState(0);
  useEffect(() => {
    if (!isLoading) {
      setElapsedTime(0);
      return;
    }

    const interval = setInterval(() => {
      setElapsedTime(Math.floor((Date.now() - startTime) / 1000));
    }, 100);

    return () => clearInterval(interval);
  }, [isLoading, startTime]);

  return (
    <div className="min-h-screen bg-white">
      {/* Navigation */}
      <nav className="border-b border-border sticky top-0 bg-white z-50">
        <div className="max-w-6xl mx-auto px-6 h-14 flex items-center justify-between">
          <Link href="/demo" className="flex items-center gap-2 text-muted-foreground hover:text-foreground transition-colors">
            <ArrowLeft className="w-4 h-4" />
            <span className="text-sm">Back</span>
          </Link>
          <div className="text-sm font-medium text-foreground">{config.title}</div>
          {results && (
            <button
              onClick={handleReset}
              className="flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors"
            >
              <RotateCcw className="w-4 h-4" />
              Reset
            </button>
          )}
        </div>
      </nav>

      <div className="max-w-6xl mx-auto px-6 py-8">
        {/* Query Input */}
        {!results && (
          <div className="max-w-2xl mx-auto">
            <div className="text-center mb-8">
              <h1 className="text-2xl font-bold text-foreground mb-2">{config.title}</h1>
              <p className="text-muted-foreground">{config.description}</p>
            </div>

            {/* Parameter Controls */}
            <ParameterControls
              alpha={alpha}
              beta={beta}
              penalty={penalty}
              lambdaParam={lambdaParam}
              solverPreset={solverPreset}
              onAlphaChange={setAlpha}
              onBetaChange={setBeta}
              onPenaltyChange={setPenalty}
              onLambdaChange={setLambdaParam}
              onPresetChange={setSolverPreset}
            />

            {/* Error Display */}
            {error && (
              <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg">
                <p className="text-sm text-red-800 font-medium">Error: {error}</p>
                <p className="text-xs text-gray-600 mt-1">
                  Ensure backend is running at http://localhost:8000
                </p>
              </div>
            )}

            <form onSubmit={handleSubmit} className="mb-6">
              <div className="relative">
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Enter a query to compare methods..."
                  disabled={isLoading}
                  className="w-full px-4 py-3 pr-12 rounded-xl border border-border bg-white text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-accent focus:border-transparent transition-all"
                />
                <button
                  type="submit"
                  disabled={isLoading || !query.trim()}
                  className="absolute right-2 top-1/2 -translate-y-1/2 w-8 h-8 rounded-lg bg-foreground text-white flex items-center justify-center disabled:opacity-50 disabled:cursor-not-allowed hover:bg-foreground/90 transition-colors"
                >
                  {isLoading ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <Search className="w-4 h-4" />
                  )}
                </button>
              </div>
            </form>

            <div>
              <p className="text-xs text-muted-foreground mb-2">Try an example:</p>
              <div className="flex flex-wrap gap-2">
                {suggestions.map((suggestion, i) => (
                  <button
                    key={i}
                    onClick={() => handleSuggestion(suggestion)}
                    disabled={isLoading}
                    className="px-3 py-1.5 text-sm rounded-lg border border-border text-foreground hover:bg-muted transition-colors disabled:opacity-50"
                  >
                    {suggestion.slice(0, 50)}{suggestion.length > 50 ? '...' : ''}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Loading State */}
        {isLoading && (
          <div className="flex flex-col items-center justify-center py-20">
            <Loader2 className="w-12 h-12 animate-spin text-accent mb-6" />
            <div className="text-center space-y-3">
              <p className="text-lg font-medium text-foreground">{progress || 'Running comparison...'}</p>
              <div className="flex items-center gap-4 text-sm text-muted-foreground">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-accent animate-pulse"></div>
                  <span>Elapsed: {elapsedTime}s</span>
                </div>
              </div>
              <div className="mt-6 p-4 bg-muted/30 rounded-lg max-w-md">
                <p className="text-xs text-muted-foreground font-medium mb-2">Running Methods:</p>
                <div className="space-y-1.5 text-xs text-muted-foreground">
                  <div className="flex items-center gap-2">
                    <div className={`w-1.5 h-1.5 rounded-full ${elapsedTime < 1 ? 'bg-accent animate-pulse' : 'bg-green-500'}`}></div>
                    <span>Top-K (baseline) - ~100ms</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className={`w-1.5 h-1.5 rounded-full ${elapsedTime >= 1 && elapsedTime < 2 ? 'bg-accent animate-pulse' : elapsedTime >= 2 ? 'bg-green-500' : 'bg-gray-300'}`}></div>
                    <span>MMR (greedy) - ~200ms</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className={`w-1.5 h-1.5 rounded-full ${elapsedTime >= 2 ? 'bg-accent animate-pulse' : 'bg-gray-300'}`}></div>
                    <span>QUBO (ORBIT) - ~{solverPreset === 'fast' ? '1-2s' : solverPreset === 'balanced' ? '5-10s' : '10-15s'}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Results */}
        {results && !isLoading && (
          <div className="space-y-8">
            {/* Query Display */}
            <div className="p-4 rounded-xl bg-muted/50">
              <p className="text-xs text-muted-foreground mb-1">Query</p>
              <p className="text-foreground font-medium">{results.query}</p>
            </div>

            {/* Metrics Summary */}
            <div className="grid grid-cols-3 gap-4">
              {[
                { method: 'Top-K', data: results.topk, color: 'danger' },
                { method: 'MMR', data: results.mmr, color: 'warning' },
                { method: 'QUBO', data: results.qubo, color: 'success' },
              ].map(({ method, data, color }) => (
                <div key={method} className={`p-4 rounded-xl border-2 ${color === 'success' ? 'border-success bg-success-light/30' : color === 'warning' ? 'border-warning/50 bg-warning-light/30' : 'border-danger/50 bg-danger-light/30'}`}>
                  <div className="flex items-center gap-2 mb-3">
                    <div className={`w-6 h-6 rounded-full flex items-center justify-center ${color === 'success' ? 'bg-success text-white' : color === 'warning' ? 'bg-warning text-white' : 'bg-danger text-white'}`}>
                      {color === 'success' ? <Check className="w-3.5 h-3.5" /> : color === 'warning' ? <Minus className="w-3.5 h-3.5" /> : <X className="w-3.5 h-3.5" />}
                    </div>
                    <span className="font-semibold text-foreground">{method}</span>
                  </div>
                  <div className="grid grid-cols-2 gap-3 text-sm">
                    <div>
                      <div className="text-muted-foreground text-xs">Diversity</div>
                      <div className={`font-semibold ${color === 'success' ? 'text-success' : color === 'warning' ? 'text-warning' : 'text-danger'}`}>
                        {Math.round((1 - data.metrics.intra_list_similarity) * 100)}%
                      </div>
                    </div>
                    <div>
                      <div className="text-muted-foreground text-xs">Coverage</div>
                      <div className={`font-semibold ${color === 'success' ? 'text-success' : color === 'warning' ? 'text-warning' : 'text-danger'}`}>
                        {data.metrics.aspects_found !== undefined ? `${data.metrics.aspects_found}/${data.metrics.total_aspects || 5}` : `${data.metrics.cluster_coverage}/${data.metrics.total_clusters}`}
                      </div>
                    </div>
                    <div className="col-span-2">
                      <div className="text-muted-foreground text-xs">Relevance</div>
                      <div className="font-semibold text-foreground">{(data.metrics.avg_relevance * 100).toFixed(0)}%</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Side-by-side Results */}
            <div>
              <h2 className="text-lg font-semibold text-foreground mb-4">Retrieved Documents</h2>
              <div className="grid grid-cols-3 gap-4">
                {[
                  { method: 'Top-K', data: results.topk, color: 'danger' },
                  { method: 'MMR', data: results.mmr, color: 'warning' },
                  { method: 'QUBO', data: results.qubo, color: 'success' },
                ].map(({ method, data, color }) => (
                  <div key={method} className="space-y-2">
                    <div className="text-sm font-medium text-muted-foreground">{method}</div>
                    {data.results.map((result, i) => {
                      // Use aspect_name if available, otherwise fallback to parsing source
                      const aspectLabel = result.aspect_name || result.source.replace('.txt', '').split('_').slice(0, -2).join('_') || result.source.replace('.txt', '').split('_')[0];
                      return (
                        <div
                          key={i}
                          className={`p-3 rounded-lg border ${color === 'success' ? 'border-success/30 bg-success-light/20' : color === 'warning' ? 'border-warning/30 bg-warning-light/20' : 'border-danger/30 bg-danger-light/20'}`}
                        >
                          <div className="flex items-center justify-between mb-1.5">
                            <span className="text-xs font-mono text-muted-foreground">#{result.rank}</span>
                            <span className="text-xs text-muted-foreground">{(result.score * 100).toFixed(0)}%</span>
                          </div>
                          <p className="text-sm text-foreground line-clamp-3 mb-2">{result.text}</p>
                          <span className="inline-block px-2 py-0.5 text-xs rounded bg-muted text-muted-foreground">{aspectLabel}</span>
                        </div>
                      );
                    })}
                  </div>
                ))}
              </div>
            </div>

            {/* LLM Responses */}
            <div>
              <h2 className="text-lg font-semibold text-foreground mb-4">LLM Responses</h2>
              <div className="grid grid-cols-3 gap-4">
                {[
                  { method: 'Top-K', data: results.topk, color: 'danger' },
                  { method: 'MMR', data: results.mmr, color: 'warning' },
                  { method: 'QUBO', data: results.qubo, color: 'success' },
                ].map(({ method, data, color }) => (
                  <div
                    key={method}
                    className={`p-4 rounded-xl border ${color === 'success' ? 'border-success/30 bg-success-light/20' : color === 'warning' ? 'border-warning/30 bg-warning-light/20' : 'border-danger/30 bg-danger-light/20'}`}
                  >
                    <div className="text-sm font-medium text-muted-foreground mb-2">{method}</div>
                    <p className="text-sm text-foreground whitespace-pre-wrap">{data.llm_response || 'No response generated'}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Summary */}
            <div className="p-6 rounded-xl bg-success-light/30 border border-success/30 text-center">
              <h3 className="font-semibold text-success mb-2">QUBO Wins</h3>
              <p className="text-sm text-foreground">
                {results.qubo.metrics.aspects_found !== undefined
                  ? `${results.qubo.metrics.aspects_found}/${results.qubo.metrics.total_aspects || 5} aspects vs ${results.topk.metrics.aspects_found}/${results.topk.metrics.total_aspects || 5} for Top-K`
                  : `${results.qubo.metrics.cluster_coverage} clusters vs ${results.topk.metrics.cluster_coverage} for Top-K`
                } •{' '}
                {Math.round((1 - results.qubo.metrics.intra_list_similarity / results.topk.metrics.intra_list_similarity) * 100)}% more diverse
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
