'use client';

import Link from 'next/link';
import { ArrowLeft, ArrowRight, Stethoscope, Scale, FlaskConical, BookOpen } from 'lucide-react';

const scenarios = [
  {
    id: 'medical',
    title: 'Medical Diagnosis',
    description: 'Differential diagnosis with overlapping symptoms. QUBO finds diverse conditions while Top-K returns duplicates.',
    icon: Stethoscope,
    badge: 'Recommended',
    stats: { docs: 210, clusters: 14 },
  },
  {
    id: 'legal',
    title: 'Legal Case Law',
    description: 'Find relevant precedents across different case types. Shows enterprise search improvement.',
    icon: Scale,
    badge: null,
    stats: { docs: 150, clusters: 5 },
  },
  {
    id: 'greedy_trap',
    title: 'Adversarial Test',
    description: 'Dataset designed to exploit MMR weaknesses. Demonstrates QUBO global optimization.',
    icon: FlaskConical,
    badge: 'Technical',
    stats: { docs: 35, clusters: 5 },
  },
  {
    id: 'wikipedia',
    title: 'Wikipedia Knowledge',
    description: 'Large-scale knowledge retrieval across 171 diverse topics. Best demonstrates diversity improvements at scale.',
    icon: BookOpen,
    badge: 'Large Scale',
    stats: { docs: 5600, clusters: 171 },
  },
];

export default function DemoPage() {
  return (
    <div className="min-h-screen bg-white">
      {/* Navigation */}
      <nav className="border-b border-border">
        <div className="max-w-5xl mx-auto px-6 h-16 flex items-center">
          <Link href="/" className="flex items-center gap-2 text-muted-foreground hover:text-foreground transition-colors">
            <ArrowLeft className="w-4 h-4" />
            <span className="text-sm">Back</span>
          </Link>
        </div>
      </nav>

      <div className="max-w-5xl mx-auto px-6 py-12">
        {/* Header */}
        <div className="mb-10">
          <h1 className="text-3xl font-bold text-foreground mb-2">Choose a scenario</h1>
          <p className="text-muted-foreground">
            Each demo shows the same comparison across different domains
          </p>
        </div>

        {/* Scenario Cards */}
        <div className="grid gap-4">
          {scenarios.map((scenario) => (
            <Link key={scenario.id} href={`/demo/${scenario.id}`}>
              <div className="group flex items-center gap-6 p-6 rounded-xl border border-border bg-white hover:border-foreground/20 hover:shadow-md transition-all cursor-pointer">
                {/* Icon */}
                <div className="w-12 h-12 rounded-xl bg-muted flex items-center justify-center flex-shrink-0 group-hover:bg-foreground group-hover:text-white transition-colors">
                  <scenario.icon className="w-6 h-6" />
                </div>

                {/* Content */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <h2 className="font-semibold text-foreground">{scenario.title}</h2>
                    {scenario.badge && (
                      <span className="px-2 py-0.5 text-xs font-medium rounded-full bg-accent-light text-accent">
                        {scenario.badge}
                      </span>
                    )}
                  </div>
                  <p className="text-sm text-muted-foreground mb-2">{scenario.description}</p>
                  <div className="flex items-center gap-4 text-xs text-muted-foreground">
                    <span>{scenario.stats.docs} documents</span>
                    <span>•</span>
                    <span>{scenario.stats.clusters} clusters</span>
                  </div>
                </div>

                {/* Arrow */}
                <ArrowRight className="w-5 h-5 text-muted-foreground group-hover:text-foreground group-hover:translate-x-1 transition-all flex-shrink-0" />
              </div>
            </Link>
          ))}
        </div>

        {/* Quick start */}
        <div className="mt-10 p-6 rounded-xl bg-muted/50 text-center">
          <p className="text-sm text-muted-foreground mb-3">
            Not sure which to pick? Start with Medical Diagnosis for the clearest demonstration.
          </p>
          <Link
            href="/demo/medical"
            className="inline-flex items-center gap-2 px-4 py-2 bg-foreground text-white rounded-lg text-sm font-medium hover:bg-foreground/90 transition-colors"
          >
            Start Medical Demo
            <ArrowRight className="w-4 h-4" />
          </Link>
        </div>
      </div>
    </div>
  );
}
