'use client';

import { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { Search, Loader2, Sparkles } from 'lucide-react';
import { cn } from '@/lib/utils';

interface QueryInputProps {
  onSubmit: (query: string) => void;
  isLoading?: boolean;
  suggestions?: string[];
  placeholder?: string;
}

export function QueryInput({
  onSubmit,
  isLoading = false,
  suggestions = [],
  placeholder = 'Enter your query...',
}: QueryInputProps) {
  const [query, setQuery] = useState('');

  const handleSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      if (query.trim() && !isLoading) {
        onSubmit(query.trim());
      }
    },
    [query, isLoading, onSubmit]
  );

  const handleSuggestionClick = useCallback(
    (suggestion: string) => {
      setQuery(suggestion);
      onSubmit(suggestion);
    },
    [onSubmit]
  );

  return (
    <div className="w-full">
      <form onSubmit={handleSubmit} className="relative">
        <div className="relative">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={placeholder}
            disabled={isLoading}
            className={cn(
              'w-full px-5 py-4 pr-14 rounded-xl border-2 text-lg',
              'bg-white shadow-lg',
              'focus:outline-none focus:ring-2 focus:ring-quantum-500 focus:border-quantum-500',
              'transition-all duration-200',
              isLoading && 'opacity-75 cursor-not-allowed'
            )}
          />
          <button
            type="submit"
            disabled={isLoading || !query.trim()}
            className={cn(
              'absolute right-2 top-1/2 -translate-y-1/2',
              'w-10 h-10 rounded-lg flex items-center justify-center',
              'bg-quantum-600 text-white',
              'hover:bg-quantum-700 transition-colors',
              'disabled:opacity-50 disabled:cursor-not-allowed'
            )}
          >
            {isLoading ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <Search className="w-5 h-5" />
            )}
          </button>
        </div>
      </form>

      {/* Suggestions */}
      {suggestions.length > 0 && !query && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-4"
        >
          <div className="flex items-center gap-2 mb-2">
            <Sparkles className="w-4 h-4 text-quantum-500" />
            <span className="text-sm text-gray-500">Try these examples:</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {suggestions.map((suggestion, i) => (
              <motion.button
                key={i}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: i * 0.05 }}
                onClick={() => handleSuggestionClick(suggestion)}
                disabled={isLoading}
                className={cn(
                  'px-3 py-2 rounded-lg text-sm',
                  'bg-quantum-50 text-quantum-700 border border-quantum-200',
                  'hover:bg-quantum-100 hover:border-quantum-300',
                  'transition-colors cursor-pointer',
                  'disabled:opacity-50 disabled:cursor-not-allowed'
                )}
              >
                {suggestion}
              </motion.button>
            ))}
          </div>
        </motion.div>
      )}
    </div>
  );
}
