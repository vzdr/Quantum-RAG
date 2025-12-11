'use client';

import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';

interface MetricsCardProps {
  title: string;
  value: string | number;
  unit?: string;
  subtitle?: string;
  color?: 'success' | 'warning' | 'danger' | 'quantum' | 'neutral';
  trend?: 'up' | 'down' | 'neutral';
  animate?: boolean;
}

const colorClasses = {
  success: 'text-success-600 bg-success-50 border-success-200',
  warning: 'text-warning-600 bg-warning-50 border-warning-200',
  danger: 'text-danger-600 bg-danger-50 border-danger-200',
  quantum: 'text-quantum-600 bg-quantum-50 border-quantum-200',
  neutral: 'text-gray-600 bg-gray-50 border-gray-200',
};

export function MetricsCard({
  title,
  value,
  unit,
  subtitle,
  color = 'neutral',
  trend,
  animate = true,
}: MetricsCardProps) {
  return (
    <motion.div
      initial={animate ? { opacity: 0, scale: 0.95 } : undefined}
      animate={animate ? { opacity: 1, scale: 1 } : undefined}
      className={cn(
        'rounded-xl p-4 border',
        colorClasses[color]
      )}
    >
      <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-1">
        {title}
      </p>
      <div className="flex items-baseline gap-1">
        <span className="text-2xl font-bold">
          {value}
        </span>
        {unit && (
          <span className="text-sm font-medium opacity-75">
            {unit}
          </span>
        )}
        {trend && (
          <span className={cn(
            'text-xs ml-1',
            trend === 'up' && 'text-success-600',
            trend === 'down' && 'text-danger-600'
          )}>
            {trend === 'up' ? '↑' : trend === 'down' ? '↓' : '–'}
          </span>
        )}
      </div>
      {subtitle && (
        <p className="text-xs text-gray-500 mt-1">{subtitle}</p>
      )}
    </motion.div>
  );
}
