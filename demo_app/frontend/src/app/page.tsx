'use client';

import Link from 'next/link';
import { ArrowRight, Check, X } from 'lucide-react';

export default function HomePage() {
  return (
    <div className="min-h-screen bg-white">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-white/80 backdrop-blur-sm border-b border-border">
        <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="font-semibold text-lg">Quantum-RAG</div>
          <Link
            href="/demo/medical"
            className="px-4 py-2 bg-foreground text-white rounded-lg text-sm font-medium hover:bg-foreground/90 transition-colors"
          >
            Try Demo
          </Link>
        </div>
      </nav>

      {/* Hero */}
      <section className="pt-32 pb-20 px-6">
        <div className="max-w-3xl mx-auto text-center">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-accent-light text-accent text-sm font-medium mb-6">
            <span className="w-2 h-2 rounded-full bg-accent animate-pulse-subtle" />
            Quantum-Inspired Retrieval
          </div>

          <h1 className="text-5xl sm:text-6xl font-bold tracking-tight text-foreground mb-6 leading-[1.1]">
            RAG that works under redundancy
          </h1>

          <p className="text-xl text-muted-foreground mb-10 leading-relaxed">
            Traditional retrieval fails when documents are similar.
            Our QUBO-based approach finds diverse, relevant context—delivering
            3x better recall and 4x token efficiency.
          </p>

          <div className="flex items-center justify-center gap-4">
            <Link
              href="/demo/medical"
              className="inline-flex items-center gap-2 px-6 py-3 bg-foreground text-white rounded-lg font-medium hover:bg-foreground/90 transition-colors"
            >
              See the difference
              <ArrowRight className="w-4 h-4" />
            </Link>
            <Link
              href="/demo"
              className="px-6 py-3 text-foreground rounded-lg font-medium hover:bg-muted transition-colors"
            >
              View all demos
            </Link>
          </div>
        </div>
      </section>

      {/* Problem/Solution Comparison */}
      <section className="py-20 px-6 bg-muted/50">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-foreground mb-3">The redundancy problem</h2>
            <p className="text-muted-foreground">
              When your documents are similar, traditional Top-K retrieval fills context with duplicates
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            {/* Top-K Card */}
            <div className="bg-white rounded-xl border border-border p-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-8 h-8 rounded-lg bg-danger-light flex items-center justify-center">
                  <X className="w-4 h-4 text-danger" />
                </div>
                <div>
                  <div className="font-semibold text-foreground">Top-K Retrieval</div>
                  <div className="text-sm text-muted-foreground">Industry standard</div>
                </div>
              </div>

              <div className="space-y-2 mb-4">
                {[1, 2, 3, 4, 5].map((i) => (
                  <div key={i} className="flex items-center gap-3 p-3 bg-danger-light/50 rounded-lg">
                    <span className="text-xs font-mono text-danger">#{i}</span>
                    <span className="text-sm text-foreground/80">Mononucleosis causes fatigue...</span>
                  </div>
                ))}
              </div>

              <div className="pt-4 border-t border-border">
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <div className="text-muted-foreground">Diversity</div>
                    <div className="font-semibold text-danger">28%</div>
                  </div>
                  <div>
                    <div className="text-muted-foreground">Conditions found</div>
                    <div className="font-semibold text-danger">1 of 5</div>
                  </div>
                </div>
              </div>
            </div>

            {/* QUBO Card */}
            <div className="bg-white rounded-xl border-2 border-success p-6 shadow-md">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-8 h-8 rounded-lg bg-success-light flex items-center justify-center">
                  <Check className="w-4 h-4 text-success" />
                </div>
                <div>
                  <div className="font-semibold text-foreground">Quantum-RAG</div>
                  <div className="text-sm text-muted-foreground">QUBO optimization</div>
                </div>
              </div>

              <div className="space-y-2 mb-4">
                {[
                  { color: 'bg-blue-100', label: 'Mononucleosis symptoms...' },
                  { color: 'bg-green-100', label: 'Lupus joint inflammation...' },
                  { color: 'bg-amber-100', label: 'Lyme disease fatigue...' },
                  { color: 'bg-purple-100', label: 'Fibromyalgia widespread pain...' },
                  { color: 'bg-pink-100', label: 'Chronic fatigue syndrome...' },
                ].map((item, i) => (
                  <div key={i} className={`flex items-center gap-3 p-3 ${item.color} rounded-lg`}>
                    <span className="text-xs font-mono text-foreground/60">#{i + 1}</span>
                    <span className="text-sm text-foreground/80">{item.label}</span>
                  </div>
                ))}
              </div>

              <div className="pt-4 border-t border-border">
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <div className="text-muted-foreground">Diversity</div>
                    <div className="font-semibold text-success">69%</div>
                  </div>
                  <div>
                    <div className="text-muted-foreground">Conditions found</div>
                    <div className="font-semibold text-success">5 of 5</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Metrics */}
      <section className="py-20 px-6">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-foreground mb-3">Measurable results</h2>
            <p className="text-muted-foreground">
              Proven improvements across key retrieval metrics
            </p>
          </div>

          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {[
              { value: '3x', label: 'Better recall', desc: 'Under redundancy' },
              { value: '4x', label: 'Token efficiency', desc: 'Cost reduction' },
              { value: '<100ms', label: 'Latency', desc: 'Production ready' },
              { value: '80%', label: 'LLM accuracy', desc: 'vs 40% Top-K' },
            ].map((metric) => (
              <div key={metric.label} className="text-center p-6 rounded-xl bg-muted/50">
                <div className="text-4xl font-bold text-foreground mb-1">{metric.value}</div>
                <div className="font-medium text-foreground">{metric.label}</div>
                <div className="text-sm text-muted-foreground">{metric.desc}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-20 px-6 bg-foreground">
        <div className="max-w-3xl mx-auto text-center">
          <h2 className="text-3xl font-bold text-white mb-4">
            See it in action
          </h2>
          <p className="text-white/70 mb-8">
            Try the interactive demo with medical or legal datasets
          </p>
          <Link
            href="/demo/medical"
            className="inline-flex items-center gap-2 px-6 py-3 bg-white text-foreground rounded-lg font-medium hover:bg-white/90 transition-colors"
          >
            Launch demo
            <ArrowRight className="w-4 h-4" />
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-8 px-6 border-t border-border">
        <div className="max-w-6xl mx-auto text-center text-sm text-muted-foreground">
          Quantum-RAG Demo • Probabilistic computing for diverse retrieval
        </div>
      </footer>
    </div>
  );
}
