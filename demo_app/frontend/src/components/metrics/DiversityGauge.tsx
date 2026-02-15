'use client';

import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';

interface DiversityGaugeProps {
  value: number; // ILS value (0-1, lower is better)
  label?: string;
  showLabel?: boolean;
}

export function DiversityGauge({ value, label = 'Diversity', showLabel = true }: DiversityGaugeProps) {
  // Convert ILS to diversity score (invert: lower ILS = higher diversity)
  const diversityScore = Math.max(0, 1 - value);
  const percentage = diversityScore * 100;

  // Color based on diversity (higher = better = greener)
  const getColor = () => {
    if (diversityScore >= 0.6) return { bg: 'bg-success-500', text: 'text-success-600' };
    if (diversityScore >= 0.45) return { bg: 'bg-warning-500', text: 'text-warning-600' };
    return { bg: 'bg-danger-500', text: 'text-danger-600' };
  };

  const colors = getColor();

  return (
    <div className="w-full">
      {showLabel && (
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm font-medium text-gray-600">{label}</span>
          <span className={cn('text-sm font-bold', colors.text)}>
            {(diversityScore * 100).toFixed(0)}%
          </span>
        </div>
      )}
      <div className="h-3 bg-gray-200 rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${percentage}%` }}
          transition={{ duration: 0.8, ease: 'easeOut' }}
          className={cn('h-full rounded-full', colors.bg)}
        />
      </div>
      <div className="flex justify-between mt-1">
        <span className="text-xs text-gray-400">Low</span>
        <span className="text-xs text-gray-400">High</span>
      </div>
    </div>
  );
}
