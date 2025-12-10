'use client';

import { useState, useCallback } from 'react';
import Link from 'next/link';
import { ArrowLeft, Search, Loader2, RotateCcw, Check, X, Minus } from 'lucide-react';
import { compareMethodsAPI, type CompareResponse } from '@/lib/api';
import { DEMO_QUERIES } from '@/lib/mockData';

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

  const handleSubmit = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || isLoading) return;

    setIsLoading(true);
    try {
      const response = await compareMethodsAPI(query.trim(), scenario, 5, true);
      setResults(response);
    } catch (error) {
      console.error('Comparison failed:', error);
    } finally {
      setIsLoading(false);
    }
  }, [query, scenario, isLoading]);

  const handleSuggestion = useCallback((suggestion: string) => {
    setQuery(suggestion);
  }, []);

  const handleReset = useCallback(() => {
    setResults(null);
    setQuery('');
  }, []);

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
            <Loader2 className="w-8 h-8 animate-spin text-muted-foreground mb-4" />
            <p className="text-muted-foreground">Running comparison...</p>
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
                        {data.metrics.cluster_coverage}/{data.metrics.total_clusters}
                      </div>
                    </div>
                    <div>
                      <div className="text-muted-foreground text-xs">Latency</div>
                      <div className="font-semibold text-foreground">{Math.round(data.metrics.latency_ms)}ms</div>
                    </div>
                    <div>
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
                      const cluster = result.source.replace('.txt', '').split('_').slice(0, -2).join('_') || result.source.replace('.txt', '').split('_')[0];
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
                          <span className="inline-block px-2 py-0.5 text-xs rounded bg-muted text-muted-foreground">{cluster}</span>
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
                {results.qubo.metrics.cluster_coverage} clusters covered vs {results.topk.metrics.cluster_coverage} for Top-K •{' '}
                {Math.round((1 - results.qubo.metrics.intra_list_similarity / results.topk.metrics.intra_list_similarity) * 100)}% more diverse •{' '}
                {Math.round(results.qubo.metrics.latency_ms)}ms latency
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
