import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Quantum-RAG Demo | Next-Gen Retrieval',
  description: 'Experience the future of RAG with quantum-inspired optimization. 3x better recall, 4x token efficiency.',
  keywords: ['RAG', 'Quantum', 'AI', 'Retrieval', 'LLM', 'Machine Learning'],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <main className="min-h-screen">
          {children}
        </main>
      </body>
    </html>
  );
}
