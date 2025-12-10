'use client';

import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';
import { RetrievedChunk } from './RetrievedChunk';
import { DiversityGauge } from '../metrics/DiversityGauge';
import { ClusterCoverage } from '../metrics/ClusterCoverage';
import type { MethodResult } from '@/lib/api';
import { Clock, Zap } from 'lucide-react';

interface MethodPanelProps {
  result: MethodResult;
  method: 'topk' | 'mmr' | 'qubo';
  title: string;
  isWinner?: boolean;
}

const methodStyles = {
  topk: {
    headerBg: 'bg-danger-600',
    headerText: 'text-white',
    icon: '❌',
    label: 'Industry Standard',
  },
  mmr: {
    headerBg: 'bg-warning-500',
    headerText: 'text-white',
    icon: '⚠️',
    label: 'Better Alternative',
  },
  qubo: {
    headerBg: 'bg-success-600',
    headerText: 'text-white',
    icon: '✨',
    label: 'Quantum-RAG',
  },
};

export function MethodPanel({ result, method, title, isWinner }: MethodPanelProps) {
  const styles = methodStyles[method];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn(
        'bg-white rounded-xl border overflow-hidden shadow-lg',
        isWinner && 'ring-2 ring-success-500 shadow-success-200'
      )}
    >
      {/* Header */}
      <div className={cn('px-4 py-3', styles.headerBg)}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-lg">{styles.icon}</span>
            <div>
              <h3 className={cn('font-bold', styles.headerText)}>{title}</h3>
              <p className={cn('text-xs opacity-80', styles.headerText)}>{styles.label}</p>
            </div>
          </div>
          <div className="flex items-center gap-2 text-white/90">
            <Clock className="w-4 h-4" />
            <span className="text-sm font-mono">{Math.round(result.metrics.latency_ms)}ms</span>
          </div>
        </div>
      </div>

      {/* Metrics */}
      <div className="p-4 border-b border-gray-100 space-y-3">
        <DiversityGauge value={result.metrics.intra_list_similarity} />
        <ClusterCoverage
          covered={result.metrics.cluster_coverage}
          total={result.metrics.total_clusters}
        />
      </div>

      {/* Results */}
      <div className="p-4 space-y-3 max-h-[400px] overflow-y-auto">
        {result.results.map((r, i) => (
          <RetrievedChunk
            key={r.chunk_id}
            result={r}
            index={i}
            method={method}
          />
        ))}
      </div>

      {/* LLM Response */}
      {result.llm_response && (
        <div className="p-4 border-t border-gray-100 bg-gray-50">
          <div className="flex items-center gap-2 mb-2">
            <Zap className="w-4 h-4 text-quantum-500" />
            <span className="text-sm font-semibold text-gray-700">LLM Response</span>
          </div>
          <p className="text-sm text-gray-600 leading-relaxed">
            {result.llm_response}
          </p>
        </div>
      )}
    </motion.div>
  );
}
