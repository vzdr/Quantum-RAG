# Quantum-RAG Investor Demo

A professional demo application showcasing Quantum-RAG's superiority over traditional retrieval methods.

## Quick Start

### 1. Start the Backend

```bash
cd demo_app/backend

# Install dependencies
pip install -r requirements.txt

# Start the API server
python main.py
```

The backend will run at `http://localhost:8000`. API docs available at `http://localhost:8000/docs`.

### 2. Start the Frontend

```bash
cd demo_app/frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

The frontend will run at `http://localhost:3000`.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FRONTEND (Next.js 14)                        │
│                    http://localhost:3000                        │
│  Landing Page → Demo Selection → Comparison View                │
└─────────────────────────────────────────────────────────────────┘
                              ↕ HTTP/REST
┌─────────────────────────────────────────────────────────────────┐
│                    BACKEND (FastAPI)                            │
│                    http://localhost:8000                        │
│  /api/compare → Runs Top-K, MMR, QUBO in parallel               │
│  /api/embeddings → UMAP coordinates for visualization           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Existing Core (../core)                      │
│  retrieval_strategies.py │ qubo_solver.py │ diversity_metrics   │
└─────────────────────────────────────────────────────────────────┘
```

## Demo Scenarios

1. **Medical Diagnosis** - Shows how QUBO finds diverse differential diagnoses
2. **Legal Case Law** - Demonstrates precedent search across case types
3. **Adversarial Test** - Dataset designed to break MMR's greedy algorithm

## Key Features

- **Split-screen comparison**: Top-K vs MMR vs QUBO side-by-side
- **Real-time metrics**: Latency, diversity (ILS), cluster coverage
- **Embedding visualization**: Interactive UMAP plot
- **LLM responses**: Gemini-powered answer generation

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/compare` | POST | Run all three methods and compare |
| `/api/retrieve` | POST | Run single method |
| `/api/embeddings/{dataset}` | GET | Get UMAP coordinates |
| `/api/datasets` | GET | List available datasets |
| `/api/health` | GET | Health check |

## Demo Script (8-10 minutes)

1. **The Problem** - Show Top-K failing on redundant data
2. **The Solution** - Reveal QUBO's diverse results
3. **The Visual** - Embedding space shows clustering vs spreading
4. **The Impact** - 80% vs 40% accuracy, 4x token savings
5. **Versatility** - Legal demo shows cross-domain applicability

## Troubleshooting

**Backend won't start:**
- Check ORBIT is installed: `pip install path/to/orbit-0.2.0-py3-none-any.whl`
- Ensure GEMINI_API_KEY is set in environment or `.env` file

**Frontend won't start:**
- Run `npm install` to ensure all dependencies are installed
- Check Node.js version is 18+

**UMAP visualization is slow:**
- First load computes UMAP (cached afterward)
- Use `/api/embeddings/{dataset}?force_recompute=false`

## Environment Variables

```bash
# Backend (.env in demo_app/backend/)
GEMINI_API_KEY=your_api_key_here

# Frontend (.env.local in demo_app/frontend/)
NEXT_PUBLIC_API_URL=http://localhost:8000
```
