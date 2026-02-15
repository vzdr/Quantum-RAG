'use client';

import { useState } from 'react';
import { Settings, Info } from 'lucide-react';

interface ParameterControlsProps {
  alpha: number;
  penalty: number;
  lambdaParam: number;
  solverPreset: string;
  onAlphaChange: (value: number) => void;
  onPenaltyChange: (value: number) => void;
  onLambdaChange: (value: number) => void;
  onPresetChange: (value: string) => void;
}

export default function ParameterControls({
  alpha,
  penalty,
  lambdaParam,
  solverPreset,
  onAlphaChange,
  onPenaltyChange,
  onLambdaChange,
  onPresetChange,
}: ParameterControlsProps) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="border border-gray-200 rounded-xl p-4 bg-gray-50 mb-6">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-2 w-full text-left"
        type="button"
      >
        <Settings className="w-4 h-4 text-gray-600" />
        <span className="text-sm font-medium text-gray-900">
          Advanced Parameters
        </span>
        <span className="ml-auto text-xs text-gray-500">
          {expanded ? '▼ Hide' : '▶ Show'}
        </span>
      </button>

      {expanded && (
        <div className="mt-4 space-y-4 pt-4 border-t border-gray-200">
          {/* Alpha - QUBO Diversity Weight */}
          <div>
            <div className="flex items-center gap-2 mb-2">
              <label className="text-sm font-medium text-gray-900">
                Alpha (QUBO Diversity)
              </label>
              <span className="ml-auto text-sm font-mono text-gray-600">
                {alpha.toFixed(3)}
              </span>
            </div>
            <input
              type="range"
              min="0"
              max="0.5"
              step="0.01"
              value={alpha}
              onChange={(e) => onAlphaChange(parseFloat(e.target.value))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-green-600"
            />
            <p className="text-xs text-gray-600 mt-1">
              Controls diversity emphasis in QUBO. Default 0.15 provides moderate diversity boost.
              Higher = more diversity, lower = more relevance. Range: 0.0-0.5.
            </p>
          </div>

          {/* Penalty - QUBO Cardinality Constraint */}
          <div>
            <div className="flex items-center gap-2 mb-2">
              <label className="text-sm font-medium text-gray-900">
                Penalty (QUBO Constraint)
              </label>
              <span className="ml-auto text-sm font-mono text-gray-600">
                {penalty.toFixed(0)}
              </span>
            </div>
            <input
              type="range"
              min="100"
              max="5000"
              step="100"
              value={penalty}
              onChange={(e) => onPenaltyChange(parseFloat(e.target.value))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-green-600"
            />
            <p className="text-xs text-gray-600 mt-1">
              Enforces selection of exactly k documents. Notebook uses 1000.
              Too low = wrong number of results, too high = numerical instability.
            </p>
          </div>

          {/* Lambda - MMR Parameter */}
          <div>
            <div className="flex items-center gap-2 mb-2">
              <label className="text-sm font-medium text-gray-900">
                Lambda (MMR Tradeoff)
              </label>
              <span className="ml-auto text-sm font-mono text-gray-600">
                {lambdaParam.toFixed(2)}
              </span>
            </div>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={lambdaParam}
              onChange={(e) => onLambdaChange(parseFloat(e.target.value))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-orange-500"
            />
            <p className="text-xs text-gray-600 mt-1">
              MMR relevance-diversity tradeoff. 0 = maximum diversity, 1 = maximum relevance.
              Standard value is 0.5 (balanced).
            </p>
          </div>

          {/* Solver Preset */}
          <div>
            <label className="block text-sm font-medium text-gray-900 mb-2">
              ORBIT Solver Preset
            </label>
            <select
              value={solverPreset}
              onChange={(e) => onPresetChange(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg bg-white text-gray-900 text-sm focus:ring-2 focus:ring-green-500 focus:border-transparent"
            >
              <option value="fast">Fast (2 replicas, 5K sweeps) - ~1s</option>
              <option value="balanced">Balanced (4 replicas, 10K sweeps) - ~2s</option>
              <option value="quality">Quality (6 replicas, 12K sweeps) - ~3s</option>
            </select>
            <p className="text-xs text-gray-600 mt-1">
              Trade-off between speed and solution quality. Notebook uses 'balanced'.
            </p>
          </div>

          {/* Info Box */}
          <div className="flex gap-2 p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <Info className="w-4 h-4 text-blue-600 flex-shrink-0 mt-0.5" />
            <p className="text-xs text-gray-900">
              These parameters match the competition notebook. Adjust to see how
              retrieval behavior changes. Reset to defaults for optimal performance.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
