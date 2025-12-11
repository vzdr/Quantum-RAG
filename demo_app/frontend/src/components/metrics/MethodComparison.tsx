'use client';

import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';
import { formatLatency, getDiversityColor, getCoverageColor, getLatencyColor } from '@/lib/utils';
import type { MethodResult } from '@/lib/api';

interface MethodComparisonProps {
  topk: MethodResult;
  mmr: MethodResult;
  qubo: MethodResult;
}

interface MetricRowProps {
  label: string;
  topkValue: string | number;
  mmrValue: string | number;
  quboValue: string | number;
  topkColor?: string;
  mmrColor?: string;
  quboColor?: string;
  highlightBest?: 'lowest' | 'highest';
}

function MetricRow({
  label,
  topkValue,
  mmrValue,
  quboValue,
  topkColor,
  mmrColor,
  quboColor,
  highlightBest,
}: MetricRowProps) {
  // Determine which is best
  const values = [
    { key: 'topk', value: typeof topkValue === 'number' ? topkValue : parseFloat(topkValue as string) },
    { key: 'mmr', value: typeof mmrValue === 'number' ? mmrValue : parseFloat(mmrValue as string) },
    { key: 'qubo', value: typeof quboValue === 'number' ? quboValue : parseFloat(quboValue as string) },
  ];

  let bestKey = '';
  if (highlightBest) {
    const sorted = [...values].sort((a, b) =>
      highlightBest === 'lowest' ? a.value - b.value : b.value - a.value
    );
    bestKey = sorted[0].key;
  }

  return (
    <div className="grid grid-cols-4 gap-4 py-3 border-b border-gray-100 last:border-0">
      <div className="text-sm font-medium text-gray-600">{label}</div>
      <div className={cn(
        'text-sm font-semibold text-center',
        topkColor || 'text-gray-900',
        bestKey === 'topk' && 'ring-2 ring-success-500 rounded px-2 bg-success-50'
      )}>
        {topkValue}
      </div>
      <div className={cn(
        'text-sm font-semibold text-center',
        mmrColor || 'text-gray-900',
        bestKey === 'mmr' && 'ring-2 ring-success-500 rounded px-2 bg-success-50'
      )}>
        {mmrValue}
      </div>
      <div className={cn(
        'text-sm font-semibold text-center',
        quboColor || 'text-gray-900',
        bestKey === 'qubo' && 'ring-2 ring-success-500 rounded px-2 bg-success-50'
      )}>
        {quboValue}
      </div>
    </div>
  );
}

export function MethodComparison({ topk, mmr, qubo }: MethodComparisonProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white rounded-xl border border-gray-200 overflow-hidden"
    >
      {/* Header */}
      <div className="grid grid-cols-4 gap-4 px-4 py-3 bg-gray-50 border-b border-gray-200">
        <div className="text-sm font-semibold text-gray-500">Metric</div>
        <div className="text-sm font-semibold text-center text-danger-600">Top-K</div>
        <div className="text-sm font-semibold text-center text-warning-600">MMR</div>
        <div className="text-sm font-semibold text-center text-success-600">QUBO</div>
      </div>

      {/* Metrics */}
      <div className="px-4">
        <MetricRow
          label="Latency"
          topkValue={formatLatency(topk.metrics.latency_ms)}
          mmrValue={formatLatency(mmr.metrics.latency_ms)}
          quboValue={formatLatency(qubo.metrics.latency_ms)}
          topkColor={getLatencyColor(topk.metrics.latency_ms)}
          mmrColor={getLatencyColor(mmr.metrics.latency_ms)}
          quboColor={getLatencyColor(qubo.metrics.latency_ms)}
          highlightBest="lowest"
        />

        <MetricRow
          label="Intra-List Similarity"
          topkValue={topk.metrics.intra_list_similarity.toFixed(3)}
          mmrValue={mmr.metrics.intra_list_similarity.toFixed(3)}
          quboValue={qubo.metrics.intra_list_similarity.toFixed(3)}
          topkColor={getDiversityColor(topk.metrics.intra_list_similarity)}
          mmrColor={getDiversityColor(mmr.metrics.intra_list_similarity)}
          quboColor={getDiversityColor(qubo.metrics.intra_list_similarity)}
          highlightBest="lowest"
        />

        <MetricRow
          label="Cluster Coverage"
          topkValue={`${topk.metrics.cluster_coverage}/${topk.metrics.total_clusters}`}
          mmrValue={`${mmr.metrics.cluster_coverage}/${mmr.metrics.total_clusters}`}
          quboValue={`${qubo.metrics.cluster_coverage}/${qubo.metrics.total_clusters}`}
          topkColor={getCoverageColor(topk.metrics.cluster_coverage, topk.metrics.total_clusters)}
          mmrColor={getCoverageColor(mmr.metrics.cluster_coverage, mmr.metrics.total_clusters)}
          quboColor={getCoverageColor(qubo.metrics.cluster_coverage, qubo.metrics.total_clusters)}
        />

        <MetricRow
          label="Avg Relevance"
          topkValue={topk.metrics.avg_relevance.toFixed(3)}
          mmrValue={mmr.metrics.avg_relevance.toFixed(3)}
          quboValue={qubo.metrics.avg_relevance.toFixed(3)}
          highlightBest="highest"
        />
      </div>

      {/* Verdict */}
      <div className="px-4 py-3 bg-success-50 border-t border-success-200">
        <p className="text-sm text-success-700 font-medium text-center">
          QUBO achieves {((1 - qubo.metrics.intra_list_similarity / topk.metrics.intra_list_similarity) * 100).toFixed(0)}% lower redundancy
          with {qubo.metrics.cluster_coverage}x more cluster coverage
        </p>
      </div>
    </motion.div>
  );
}
