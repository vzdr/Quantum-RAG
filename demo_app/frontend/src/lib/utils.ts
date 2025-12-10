import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

/**
 * Merge Tailwind CSS classes with clsx
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/**
 * Format a number with specified decimal places
 */
export function formatNumber(num: number, decimals: number = 2): string {
  return num.toFixed(decimals);
}

/**
 * Format latency in milliseconds
 */
export function formatLatency(ms: number): string {
  if (ms < 1) {
    return '<1ms';
  }
  return `${Math.round(ms)}ms`;
}

/**
 * Get color class based on diversity score (ILS)
 * Lower ILS = more diverse = better = green
 */
export function getDiversityColor(ils: number): string {
  if (ils < 0.4) return 'text-success-500';
  if (ils < 0.55) return 'text-warning-500';
  return 'text-danger-500';
}

/**
 * Get color class based on cluster coverage
 * Higher coverage = better = green
 */
export function getCoverageColor(covered: number, total: number): string {
  const ratio = covered / total;
  if (ratio >= 0.8) return 'text-success-500';
  if (ratio >= 0.5) return 'text-warning-500';
  return 'text-danger-500';
}

/**
 * Get color class based on latency
 * Lower latency = better = green
 */
export function getLatencyColor(ms: number): string {
  if (ms < 100) return 'text-success-500';
  if (ms < 500) return 'text-warning-500';
  return 'text-danger-500';
}

/**
 * Generate distinct colors for clusters
 */
export function getClusterColors(clusters: string[]): Record<string, string> {
  const colors = [
    '#6366f1', // indigo
    '#10b981', // emerald
    '#f59e0b', // amber
    '#ef4444', // red
    '#8b5cf6', // violet
    '#06b6d4', // cyan
    '#f97316', // orange
    '#ec4899', // pink
    '#14b8a6', // teal
    '#84cc16', // lime
    '#a855f7', // purple
    '#0ea5e9', // sky
  ];

  const colorMap: Record<string, string> = {};
  clusters.forEach((cluster, i) => {
    colorMap[cluster] = colors[i % colors.length];
  });

  return colorMap;
}

/**
 * Truncate text to specified length
 */
export function truncateText(text: string, maxLength: number = 150): string {
  if (text.length <= maxLength) return text;
  return text.slice(0, maxLength) + '...';
}
