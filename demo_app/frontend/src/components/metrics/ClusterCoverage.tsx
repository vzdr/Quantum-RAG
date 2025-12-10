'use client';

import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';

interface ClusterCoverageProps {
  covered: number;
  total: number;
  label?: string;
}

export function ClusterCoverage({ covered, total, label = 'Cluster Coverage' }: ClusterCoverageProps) {
  const ratio = total > 0 ? covered / total : 0;

  // Color based on coverage ratio
  const getColor = () => {
    if (ratio >= 0.8) return 'bg-success-500';
    if (ratio >= 0.5) return 'bg-warning-500';
    return 'bg-danger-500';
  };

  const getTextColor = () => {
    if (ratio >= 0.8) return 'text-success-600';
    if (ratio >= 0.5) return 'text-warning-600';
    return 'text-danger-600';
  };

  return (
    <div className="w-full">
      <div className="flex justify-between items-center mb-2">
        <span className="text-sm font-medium text-gray-600">{label}</span>
        <span className={cn('text-sm font-bold', getTextColor())}>
          {covered}/{total}
        </span>
      </div>
      <div className="flex gap-1">
        {Array.from({ length: total }).map((_, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, scale: 0 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: i * 0.1 }}
            className={cn(
              'flex-1 h-4 rounded',
              i < covered ? getColor() : 'bg-gray-200'
            )}
          />
        ))}
      </div>
    </div>
  );
}
