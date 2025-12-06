"""
Visualization Module

Provides visualization functions for:
- Embedding space (UMAP/t-SNE/PCA)
- Similarity scores
- Chunk statistics
- Retrieval results
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.chunker import Chunk
from core.retriever import RetrievalResult


def create_embedding_visualization(
    embeddings: np.ndarray,
    metadata: List[Dict[str, Any]],
    method: str = 'UMAP',
    query_embedding: Optional[np.ndarray] = None,
    retrieved_indices: Optional[List[int]] = None,
    retrieval_scores: Optional[List[float]] = None,
    perplexity: int = 30,
    n_neighbors: int = 15,
    max_points: int = 2000  # Subsample if more than this
) -> go.Figure:
    """
    Create an interactive embedding visualization.

    Args:
        embeddings: Array of embeddings (n_samples, embedding_dim)
        metadata: List of metadata dicts with 'source', 'text' keys
        method: Projection method ('UMAP', 't-SNE', 'PCA')
        query_embedding: Optional query embedding to highlight
        retrieved_indices: Optional list of retrieved chunk indices
        retrieval_scores: Optional list of similarity scores for retrieved chunks
        perplexity: t-SNE perplexity parameter
        n_neighbors: UMAP n_neighbors parameter
        max_points: Subsample to this many points for speed (default 2000)

    Returns:
        Plotly figure
    """
    if len(embeddings) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No embeddings to visualize", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    # Subsample if too many points (for speed)
    if len(embeddings) > max_points:
        indices = np.random.choice(len(embeddings), max_points, replace=False)
        embeddings = embeddings[indices]
        metadata = [metadata[i] for i in indices]
        # Adjust retrieved_indices if present
        if retrieved_indices is not None:
            idx_set = set(indices)
            retrieved_indices = [i for i in retrieved_indices if i in idx_set]

    # Combine embeddings if query is provided
    all_embeddings = embeddings
    if query_embedding is not None:
        all_embeddings = np.vstack([embeddings, query_embedding.reshape(1, -1)])

    # Dimensionality reduction
    if method == 'UMAP':
        from umap import UMAP
        # Fast UMAP params: fewer neighbors, lower min_dist, no verbose
        n_neighbors_fast = min(5, len(all_embeddings)-1)  # Reduced from 15 to 5 for speed
        reducer = UMAP(
            n_neighbors=n_neighbors_fast,
            min_dist=0.1,
            metric='euclidean',  # Faster than cosine
            n_epochs=200,  # Reduced from default 500
            init='spectral',  # Faster initialization
            random_state=42,
            verbose=False,
            low_memory=False  # Use more memory for speed
        )
        reduced = reducer.fit_transform(all_embeddings)
    elif method == 't-SNE':
        from sklearn.manifold import TSNE
        perp = min(perplexity, len(all_embeddings) - 1)
        reducer = TSNE(n_components=2, perplexity=perp, random_state=42)
        reduced = reducer.fit_transform(all_embeddings)
    else:  # PCA
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)
        reduced = reducer.fit_transform(all_embeddings)

    # Separate query point if present and compute ALL similarities
    all_similarities = None
    if query_embedding is not None:
        query_reduced = reduced[-1:]
        reduced = reduced[:-1]

        # Compute cosine similarity between query and ALL embeddings
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        embedding_norms = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        all_similarities = np.dot(embedding_norms, query_norm)
        # Normalize so max similarity = 1.0 (makes colors more meaningful)
        max_sim = np.max(all_similarities)
        min_sim = np.min(all_similarities)
        if max_sim > min_sim:
            all_similarities = (all_similarities - min_sim) / (max_sim - min_sim)
        else:
            all_similarities = np.ones_like(all_similarities)

    # Create DataFrame for plotting
    df = pd.DataFrame({
        'x': reduced[:, 0],
        'y': reduced[:, 1],
        'source': [m.get('source', 'unknown') for m in metadata],
        'text': [m.get('text', '')[:100] + '...' if len(m.get('text', '')) > 100 else m.get('text', '') for m in metadata],
        'id': [m.get('id', str(i)) for i, m in enumerate(metadata)]
    })

    # Add similarity scores if available
    if all_similarities is not None:
        df['similarity'] = all_similarities

    # Create scatter plot - color by similarity if query provided, else by source
    if all_similarities is not None:
        fig = go.Figure()

        # All points colored by similarity
        fig.add_trace(go.Scatter(
            x=df['x'],
            y=df['y'],
            mode='markers',
            marker=dict(
                size=8,
                color=all_similarities,
                colorscale='RdYlGn',  # Red (low) -> Yellow -> Green (high)
                cmin=0,
                cmax=1,
                colorbar=dict(title='Similarity', x=1.02)
            ),
            text=[f"Score: {s:.3f}<br>Source: {src}<br>{txt}"
                  for s, src, txt in zip(all_similarities, df['source'], df['text'])],
            hoverinfo='text',
            name='All Chunks'
        ))

        fig.update_layout(
            title=f'Embedding Space Visualization ({method}) - Colored by Query Similarity',
            template='plotly_white'
        )
    else:
        fig = px.scatter(
            df, x='x', y='y',
            color='source',
            hover_data=['text', 'id'],
            title=f'Embedding Space Visualization ({method})',
            template='plotly_white'
        )

    # Highlight retrieved chunks with star markers
    if retrieved_indices:
        # Get coordinates for retrieved chunks
        retrieved_x = [reduced[i, 0] for i in retrieved_indices]
        retrieved_y = [reduced[i, 1] for i in retrieved_indices]

        # Get scores from all_similarities or provided scores
        if all_similarities is not None:
            scores = [all_similarities[i] for i in retrieved_indices]
        elif retrieval_scores:
            scores = retrieval_scores
        else:
            scores = [0.5] * len(retrieved_indices)

        # Get hover text
        hover_texts = [f"RETRIEVED #{rank+1}<br>Score: {s:.3f}<br>{metadata[i].get('text', '')[:100]}..."
                       for rank, (i, s) in enumerate(zip(retrieved_indices, scores))]

        fig.add_trace(go.Scatter(
            x=retrieved_x,
            y=retrieved_y,
            mode='markers',
            marker=dict(
                size=18,
                symbol='star',
                color=scores,
                colorscale='RdYlGn',
                cmin=0,
                cmax=1,
                line=dict(width=2, color='black')
            ),
            name='Retrieved',
            hovertext=hover_texts,
            hoverinfo='text'
        ))

    # Add query point
    if query_embedding is not None:
        fig.add_trace(go.Scatter(
            x=[query_reduced[0, 0]],
            y=[query_reduced[0, 1]],
            mode='markers',
            marker=dict(
                size=20,
                symbol='x',
                color='black',
                line=dict(width=3)
            ),
            name='Query'
        ))

    fig.update_layout(
        height=600,
        width=900,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis_title=f'{method} Dimension 1',
        yaxis_title=f'{method} Dimension 2'
    )

    return fig


def create_similarity_chart(results: List[RetrievalResult]) -> go.Figure:
    """
    Create a horizontal bar chart of similarity scores.

    Args:
        results: List of RetrievalResult objects

    Returns:
        Plotly figure
    """
    if not results:
        fig = go.Figure()
        fig.add_annotation(text="No results to display", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    labels = [f"Chunk {r.rank}: {r.source[:20]}..." if len(r.source) > 20 else f"Chunk {r.rank}: {r.source}" for r in results]
    scores = [r.score for r in results]
    texts = [r.text[:80] + '...' if len(r.text) > 80 else r.text for r in results]

    # Color gradient from red (low) to green (high)
    colors = [f'rgb({int(255 * (1 - s))}, {int(200 * s)}, {int(100)})' for s in scores]

    fig = go.Figure(go.Bar(
        y=labels,
        x=scores,
        orientation='h',
        marker_color=colors,
        text=[f'{s:.4f}' for s in scores],
        textposition='outside',
        hovertext=texts,
        hoverinfo='text'
    ))

    fig.update_layout(
        title='Similarity Scores',
        xaxis_title='Cosine Similarity',
        yaxis_title='Chunk',
        xaxis=dict(range=[0, max(scores) * 1.15] if scores else [0, 1]),
        height=max(250, 50 + len(results) * 40),
        template='plotly_white',
        margin=dict(l=150)
    )

    return fig


def create_chunk_statistics_dashboard(chunks: List[Chunk]) -> go.Figure:
    """
    Create a dashboard with chunk statistics.

    Args:
        chunks: List of Chunk objects

    Returns:
        Plotly figure with subplots
    """
    if not chunks:
        fig = go.Figure()
        fig.add_annotation(text="No chunks to analyze", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    lengths = [len(c.text) for c in chunks]
    word_counts = [c.word_count for c in chunks]
    sources = [c.source for c in chunks]

    # Count chunks per source
    source_counts = {}
    for s in sources:
        source_counts[s] = source_counts.get(s, 0) + 1

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Chunk Length Distribution (chars)',
            'Chunks per Document',
            'Word Count Distribution',
            'Length Statistics'
        ],
        specs=[[{"type": "histogram"}, {"type": "bar"}],
               [{"type": "histogram"}, {"type": "box"}]]
    )

    # Length histogram
    fig.add_trace(
        go.Histogram(x=lengths, nbinsx=25, name='Characters', marker_color='steelblue'),
        row=1, col=1
    )

    # Chunks per document
    fig.add_trace(
        go.Bar(
            x=list(source_counts.keys()),
            y=list(source_counts.values()),
            name='Documents',
            marker_color='coral'
        ),
        row=1, col=2
    )

    # Word count histogram
    fig.add_trace(
        go.Histogram(x=word_counts, nbinsx=20, name='Words', marker_color='mediumseagreen'),
        row=2, col=1
    )

    # Length box plot
    fig.add_trace(
        go.Box(y=lengths, name='Length Stats', marker_color='mediumpurple'),
        row=2, col=2
    )

    fig.update_layout(
        height=600,
        showlegend=False,
        template='plotly_white',
        title_text='Chunk Statistics Dashboard'
    )

    return fig


def create_retrieval_results_display(
    results: List[RetrievalResult],
    query: str
) -> str:
    """
    Create HTML display of retrieval results.

    Args:
        results: List of RetrievalResult objects
        query: The original query

    Returns:
        HTML string
    """
    if not results:
        return '<p><i>No results found</i></p>'

    html_parts = [
        f'<div style="padding: 10px; background: #f5f5f5; border-radius: 5px; margin-bottom: 15px;">',
        f'<b>Query:</b> {query}<br>',
        f'<b>Results:</b> {len(results)} chunks retrieved',
        '</div>'
    ]

    for result in results:
        score_color = '#4CAF50' if result.score > 0.7 else '#FF9800' if result.score > 0.4 else '#F44336'

        html_parts.append(f'''
        <div style="padding: 15px; border: 1px solid #ddd; border-radius: 5px; margin-bottom: 10px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                <span><b>Rank {result.rank}</b> - {result.source}</span>
                <span style="background: {score_color}; color: white; padding: 2px 8px; border-radius: 3px;">
                    Score: {result.score:.4f}
                </span>
            </div>
            <div style="color: #555; font-size: 0.95em;">
                {result.text[:300]}{'...' if len(result.text) > 300 else ''}
            </div>
        </div>
        ''')

    return ''.join(html_parts)


def create_token_usage_chart(generations: List[Dict[str, Any]]) -> go.Figure:
    """
    Create a chart showing token usage over queries.

    Args:
        generations: List of generation metadata dicts

    Returns:
        Plotly figure
    """
    if not generations:
        fig = go.Figure()
        fig.add_annotation(text="No generation data", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    queries = [g.get('query', f'Query {i+1}')[:30] + '...' for i, g in enumerate(generations)]
    prompt_tokens = [g.get('prompt_tokens', 0) for g in generations]
    completion_tokens = [g.get('completion_tokens', 0) for g in generations]

    fig = go.Figure(data=[
        go.Bar(name='Prompt Tokens', x=queries, y=prompt_tokens, marker_color='steelblue'),
        go.Bar(name='Completion Tokens', x=queries, y=completion_tokens, marker_color='coral')
    ])

    fig.update_layout(
        barmode='stack',
        title='Token Usage per Query',
        xaxis_title='Query',
        yaxis_title='Tokens',
        template='plotly_white',
        height=400
    )

    return fig


def get_statistics_summary(chunks: List[Chunk]) -> Dict[str, Any]:
    """
    Calculate summary statistics for chunks.

    Args:
        chunks: List of Chunk objects

    Returns:
        Dictionary with statistics
    """
    if not chunks:
        return {
            'total_chunks': 0,
            'total_chars': 0,
            'total_words': 0,
            'avg_length': 0,
            'min_length': 0,
            'max_length': 0,
            'avg_words': 0,
            'unique_sources': 0,
        }

    lengths = [len(c.text) for c in chunks]
    word_counts = [c.word_count for c in chunks]
    sources = set(c.source for c in chunks)

    return {
        'total_chunks': len(chunks),
        'total_chars': sum(lengths),
        'total_words': sum(word_counts),
        'avg_length': sum(lengths) / len(lengths),
        'min_length': min(lengths),
        'max_length': max(lengths),
        'avg_words': sum(word_counts) / len(word_counts),
        'unique_sources': len(sources),
        'sources': list(sources),
    }
