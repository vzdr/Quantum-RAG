'use client';

import { useMemo } from 'react';
import dynamic from 'next/dynamic';
import { getClusterColors } from '@/lib/utils';
import type { UMAPPoint } from '@/lib/api';

// Dynamic import for Plotly to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface EmbeddingSpaceProps {
  points: UMAPPoint[];
  queryPoint?: { x: number; y: number } | null;
  title?: string;
  height?: number;
}

export function EmbeddingSpace({
  points,
  queryPoint,
  title = 'Embedding Space (UMAP)',
  height = 400,
}: EmbeddingSpaceProps) {
  // Get unique clusters and assign colors
  const clusters = useMemo(() => {
    return [...new Set(points.map((p) => p.cluster))];
  }, [points]);

  const clusterColors = useMemo(() => getClusterColors(clusters), [clusters]);

  // Prepare plot data
  const plotData = useMemo(() => {
    const traces: any[] = [];

    // Group points by cluster
    clusters.forEach((cluster) => {
      const clusterPoints = points.filter((p) => p.cluster === cluster);

      // Unselected points (smaller, transparent)
      const unselected = clusterPoints.filter((p) => !p.is_selected);
      if (unselected.length > 0) {
        traces.push({
          type: 'scatter',
          mode: 'markers',
          name: cluster,
          x: unselected.map((p) => p.x),
          y: unselected.map((p) => p.y),
          marker: {
            color: clusterColors[cluster],
            size: 8,
            opacity: 0.4,
          },
          text: unselected.map((p) => `${p.source}<br>Cluster: ${p.cluster}`),
          hoverinfo: 'text',
        });
      }

      // Selected points (larger, more visible)
      const selected = clusterPoints.filter((p) => p.is_selected);
      if (selected.length > 0) {
        traces.push({
          type: 'scatter',
          mode: 'markers',
          name: `${cluster} (selected)`,
          x: selected.map((p) => p.x),
          y: selected.map((p) => p.y),
          marker: {
            color: clusterColors[cluster],
            size: 16,
            opacity: 1,
            line: {
              color: 'white',
              width: 2,
            },
          },
          text: selected.map(
            (p) =>
              `${p.source}<br>Cluster: ${p.cluster}<br>Selected by: ${p.selected_by.join(', ')}`
          ),
          hoverinfo: 'text',
        });
      }
    });

    // Add query point if available
    if (queryPoint) {
      traces.push({
        type: 'scatter',
        mode: 'markers',
        name: 'Query',
        x: [queryPoint.x],
        y: [queryPoint.y],
        marker: {
          color: '#6366f1',
          size: 20,
          symbol: 'star',
          line: {
            color: 'white',
            width: 2,
          },
        },
        text: ['Query'],
        hoverinfo: 'text',
      });
    }

    return traces;
  }, [points, clusters, clusterColors, queryPoint]);

  const layout = useMemo(
    () => ({
      title: {
        text: title,
        font: { size: 14, color: '#374151' },
      },
      showlegend: true,
      legend: {
        orientation: 'h' as const,
        y: -0.15,
        x: 0.5,
        xanchor: 'center' as const,
      },
      hovermode: 'closest' as const,
      xaxis: {
        showgrid: false,
        zeroline: false,
        showticklabels: false,
        title: '',
      },
      yaxis: {
        showgrid: false,
        zeroline: false,
        showticklabels: false,
        title: '',
      },
      margin: { l: 20, r: 20, t: 40, b: 60 },
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'rgba(249, 250, 251, 0.5)',
    }),
    [title]
  );

  const config = useMemo(
    () => ({
      displayModeBar: false,
      responsive: true,
    }),
    []
  );

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-4">
      <Plot
        data={plotData}
        layout={layout}
        config={config}
        style={{ width: '100%', height }}
      />
      <div className="mt-2 text-xs text-gray-500 text-center">
        Larger points = selected by retrieval methods. Star = query position.
      </div>
    </div>
  );
}
