"""
Chunking Widget - Phase A, Step 2 UI

Interactive chunking configuration with live preview.
"""
from typing import List, Dict, Any, Callable, Optional

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.document_loader import Document
from core.chunker import TextChunker, Chunk


class ChunkingWidget:
    """
    Interactive chunking parameter configuration.

    Features:
    - Adjustable chunk size slider
    - Overlap control
    - Strategy selection
    - Live preview of chunks
    - Statistics dashboard
    """

    def __init__(
        self,
        on_chunks_created: Optional[Callable[[List[Chunk]], None]] = None
    ):
        """
        Initialize the chunking widget.

        Args:
            on_chunks_created: Callback when chunks are created
        """
        self.documents: List[Document] = []
        self.chunks: List[Chunk] = []
        self.on_chunks_created = on_chunks_created

        self._create_widgets()
        self._bind_events()

    def _create_widgets(self):
        """Create all UI components."""
        # Chunk size slider
        self.chunk_size_slider = widgets.IntSlider(
            value=500,
            min=100,
            max=2000,
            step=50,
            description='Chunk Size:',
            continuous_update=False,
            style={'description_width': '100px'},
            layout=widgets.Layout(width='400px')
        )

        # Overlap slider
        self.overlap_slider = widgets.IntSlider(
            value=50,
            min=0,
            max=200,
            step=10,
            description='Overlap:',
            continuous_update=False,
            style={'description_width': '100px'},
            layout=widgets.Layout(width='400px')
        )

        # Strategy dropdown
        self.strategy_dropdown = widgets.Dropdown(
            options=[
                ('Fixed Size', 'fixed'),
                ('Sentence-Aware', 'sentence'),
                ('Paragraph-Aware', 'paragraph')
            ],
            value='sentence',
            description='Strategy:',
            style={'description_width': '100px'}
        )

        # Chunk button
        self.chunk_btn = widgets.Button(
            description='Create Chunks',
            button_style='success',
            icon='scissors'
        )

        # Preview output
        self.preview_output = widgets.Output()

        # Statistics output
        self.stats_output = widgets.Output()

        # Status output
        self.status_output = widgets.Output()

        # Current values display
        self.params_html = widgets.HTML()
        self._update_params_display()

    def _bind_events(self):
        """Bind widget events."""
        self.chunk_btn.on_click(self._on_chunk)
        self.chunk_size_slider.observe(self._on_param_change, names='value')
        self.overlap_slider.observe(self._on_param_change, names='value')
        self.strategy_dropdown.observe(self._on_param_change, names='value')

    def _on_param_change(self, change):
        """Handle parameter changes."""
        self._update_params_display()

    def _update_params_display(self):
        """Update the parameters display."""
        self.params_html.value = f"""
        <div style="padding: 10px; background: #f0f0f0; border-radius: 5px; margin: 10px 0;">
            <b>Current Settings:</b><br>
            Chunk Size: {self.chunk_size_slider.value} chars |
            Overlap: {self.overlap_slider.value} chars |
            Strategy: {self.strategy_dropdown.value}
        </div>
        """

    def _on_chunk(self, btn):
        """Handle chunk button click."""
        if not self.documents:
            with self.status_output:
                clear_output()
                print("No documents loaded. Please upload documents first.")
            return

        with self.status_output:
            clear_output()
            print("Creating chunks...")

        # Create chunker
        chunker = TextChunker(
            chunk_size=self.chunk_size_slider.value,
            overlap=self.overlap_slider.value,
            strategy=self.strategy_dropdown.value
        )

        # Chunk all documents
        self.chunks = chunker.chunk_documents(self.documents)

        with self.status_output:
            print(f"Created {len(self.chunks)} chunks from {len(self.documents)} documents")

        # Update displays
        self._update_preview()
        self._update_statistics()

        # Callback
        if self.on_chunks_created:
            self.on_chunks_created(self.chunks)

    def _update_preview(self):
        """Update chunk preview."""
        with self.preview_output:
            clear_output()

            if not self.chunks:
                print("No chunks created yet")
                return

            # Show first 5 chunks
            preview_count = min(5, len(self.chunks))
            print(f"Showing first {preview_count} of {len(self.chunks)} chunks:\n")

            for i, chunk in enumerate(self.chunks[:preview_count]):
                preview = chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
                print(f"--- Chunk {i + 1} [{chunk.source}] ({len(chunk)} chars) ---")
                print(preview)
                print()

    def _update_statistics(self):
        """Update statistics visualization."""
        with self.stats_output:
            clear_output()

            if not self.chunks:
                print("No chunks to analyze")
                return

            # Calculate statistics
            lengths = [len(c.text) for c in self.chunks]
            word_counts = [c.word_count for c in self.chunks]
            sources = [c.source for c in self.chunks]

            # Create subplots
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=['Chunk Length Distribution', 'Chunks per Document']
            )

            # Length histogram
            fig.add_trace(
                go.Histogram(x=lengths, nbinsx=20, name='Length', marker_color='steelblue'),
                row=1, col=1
            )

            # Source bar chart
            source_counts = {}
            for s in sources:
                source_counts[s] = source_counts.get(s, 0) + 1

            fig.add_trace(
                go.Bar(
                    x=list(source_counts.keys()),
                    y=list(source_counts.values()),
                    name='Documents',
                    marker_color='coral'
                ),
                row=1, col=2
            )

            fig.update_layout(
                height=300,
                showlegend=False,
                margin=dict(l=50, r=50, t=50, b=50)
            )

            fig.show()

            # Print summary statistics
            print(f"\nStatistics Summary:")
            print(f"  Total Chunks: {len(self.chunks)}")
            print(f"  Avg Length: {sum(lengths)/len(lengths):.0f} chars")
            print(f"  Min Length: {min(lengths)} chars")
            print(f"  Max Length: {max(lengths)} chars")
            print(f"  Avg Words: {sum(word_counts)/len(word_counts):.0f}")

    def set_documents(self, documents: List[Document]):
        """Set documents to chunk."""
        self.documents = documents
        self.chunks = []

        with self.status_output:
            clear_output()
            print(f"Loaded {len(documents)} documents ready for chunking")

    def get_chunks(self) -> List[Chunk]:
        """Get created chunks."""
        return self.chunks

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return {
            'chunk_size': self.chunk_size_slider.value,
            'overlap': self.overlap_slider.value,
            'strategy': self.strategy_dropdown.value,
        }

    def display(self):
        """Display the widget."""
        title = widgets.HTML('<h3>Step 2: Text Chunking</h3>')

        controls = widgets.VBox([
            self.chunk_size_slider,
            self.overlap_slider,
            self.strategy_dropdown,
            self.chunk_btn,
        ])

        layout = widgets.VBox([
            title,
            controls,
            self.params_html,
            self.status_output,
            widgets.HTML('<h4>Chunk Preview:</h4>'),
            self.preview_output,
            widgets.HTML('<h4>Statistics:</h4>'),
            self.stats_output,
        ])

        display(layout)
        return layout
