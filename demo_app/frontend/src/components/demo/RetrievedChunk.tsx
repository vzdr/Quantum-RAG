'use client';

import { motion } from 'framer-motion';
import { cn, truncateText } from '@/lib/utils';
import type { RetrievalResult } from '@/lib/api';

interface RetrievedChunkProps {
  result: RetrievalResult;
  index: number;
  method: 'topk' | 'mmr' | 'qubo';
  isHighlighted?: boolean;
}

const methodColors = {
  topk: {
    bg: 'bg-danger-50',
    border: 'border-danger-200',
    badge: 'bg-danger-100 text-danger-700',
  },
  mmr: {
    bg: 'bg-warning-50',
    border: 'border-warning-200',
    badge: 'bg-warning-100 text-warning-700',
  },
  qubo: {
    bg: 'bg-success-50',
    border: 'border-success-200',
    badge: 'bg-success-100 text-success-700',
  },
};

export function RetrievedChunk({ result, index, method, isHighlighted }: RetrievedChunkProps) {
  const colors = methodColors[method];

  // Extract cluster/disease from source filename
  const cluster = result.source.replace('.txt', '').split('_').slice(0, -1).join('_');

  return (
    <motion.div
      initial={{ opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.1 }}
      className={cn(
        'rounded-lg border p-3 transition-all',
        colors.bg,
        colors.border,
        isHighlighted && 'ring-2 ring-quantum-500 shadow-lg'
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className={cn('px-2 py-0.5 rounded text-xs font-semibold', colors.badge)}>
            #{result.rank}
          </span>
          <span className="text-xs text-gray-500 truncate max-w-[120px]" title={result.source}>
            {result.source}
          </span>
        </div>
        <span className="text-xs font-mono text-gray-500">
          {(result.score * 100).toFixed(1)}%
        </span>
      </div>

      {/* Content */}
      <p className="text-sm text-gray-700 leading-relaxed">
        {truncateText(result.text, 200)}
      </p>

      {/* Cluster badge */}
      <div className="mt-2">
        <span className="text-xs px-2 py-0.5 rounded-full bg-gray-100 text-gray-600">
          {cluster}
        </span>
      </div>
    </motion.div>
  );
}
