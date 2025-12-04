"""
Embedding Widget - Phase A, Steps 3-4 UI

Interactive embedding generation and vector store configuration.
"""
from typing import List, Dict, Any, Callable, Optional

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

from core.chunker import Chunk
from core.embedder import EmbeddingGenerator, EmbeddedChunk
from core.vector_store import VectorStore


class EmbeddingWidget:
    """
    Interactive embedding and vector store configuration.

    Features:
    - Model selection
    - Batch size configuration
    - Progress tracking
    - Vector store management
    """

    def __init__(
        self,
        on_index_built: Optional[Callable[[VectorStore], None]] = None
    ):
        """
        Initialize the embedding widget.

        Args:
            on_index_built: Callback when index is built
        """
        self.chunks: List[Chunk] = []
        self.embedded_chunks: List[EmbeddedChunk] = []
        self.embedder: Optional[EmbeddingGenerator] = None
        self.vector_store: Optional[VectorStore] = None
        self.on_index_built = on_index_built

        self._create_widgets()
        self._bind_events()

    def _create_widgets(self):
        """Create all UI components."""
        # Model selection
        self.model_dropdown = widgets.Dropdown(
            options=[
                ('all-MiniLM-L6-v2 (Fast, 384d)', 'all-MiniLM-L6-v2'),
                ('all-mpnet-base-v2 (Balanced, 768d)', 'all-mpnet-base-v2'),
                ('multi-qa-mpnet-base-cos-v1 (QA)', 'multi-qa-mpnet-base-cos-v1'),
            ],
            value='all-MiniLM-L6-v2',
            description='Model:',
            style={'description_width': '100px'},
            layout=widgets.Layout(width='400px')
        )

        # Batch size
        self.batch_slider = widgets.IntSlider(
            value=32,
            min=8,
            max=128,
            step=8,
            description='Batch Size:',
            style={'description_width': '100px'},
            layout=widgets.Layout(width='300px')
        )

        # Device selection
        self.device_toggle = widgets.ToggleButtons(
            options=['cpu', 'cuda'],
            value='cpu',
            description='Device:',
            style={'description_width': '100px'}
        )

        # Collection name
        self.collection_input = widgets.Text(
            value='rag_collection',
            description='Collection:',
            style={'description_width': '100px'},
            layout=widgets.Layout(width='300px')
        )

        # Reset collection checkbox
        self.reset_checkbox = widgets.Checkbox(
            value=False,
            description='Reset existing collection',
            style={'description_width': 'initial'}
        )

        # Build button
        self.build_btn = widgets.Button(
            description='Build Index',
            button_style='success',
            icon='database'
        )

        # Progress bar
        self.progress_bar = widgets.IntProgress(
            value=0,
            min=0,
            max=100,
            description='Progress:',
            bar_style='info',
            layout=widgets.Layout(width='400px')
        )

        # Status output
        self.status_output = widgets.Output()

        # Statistics display
        self.stats_html = widgets.HTML()

    def _bind_events(self):
        """Bind widget events."""
        self.build_btn.on_click(self._on_build)

    def _on_build(self, btn):
        """Handle build button click."""
        if not self.chunks:
            with self.status_output:
                clear_output()
                print("No chunks available. Please create chunks first.")
            return

        with self.status_output:
            clear_output()
            print("Initializing embedding model...")

        self.progress_bar.value = 0

        try:
            # Initialize embedder
            self.embedder = EmbeddingGenerator(
                model_name=self.model_dropdown.value,
                device=self.device_toggle.value
            )

            with self.status_output:
                print(f"Model loaded: {self.model_dropdown.value}")
                print(f"Embedding dimension: {self.embedder.embedding_dim}")
                print("\nGenerating embeddings...")

            # Progress callback
            def progress_callback(current, total):
                self.progress_bar.value = int((current / total) * 100)

            # Generate embeddings
            self.embedded_chunks = self.embedder.embed_chunks(
                self.chunks,
                batch_size=self.batch_slider.value,
                show_progress=False,
                progress_callback=progress_callback
            )

            with self.status_output:
                print(f"Generated {len(self.embedded_chunks)} embeddings")
                print("\nCreating vector store...")

            # Initialize vector store
            self.vector_store = VectorStore(
                collection_name=self.collection_input.value,
                reset=self.reset_checkbox.value
            )

            # Add embeddings
            count = self.vector_store.add(self.embedded_chunks)

            self.progress_bar.value = 100

            with self.status_output:
                print(f"Added {count} vectors to collection '{self.collection_input.value}'")
                print("\nIndex built successfully!")

            # Update statistics
            self._update_statistics()

            # Callback
            if self.on_index_built and self.vector_store:
                self.on_index_built(self.vector_store)

        except Exception as e:
            with self.status_output:
                print(f"\nError: {e}")
            self.progress_bar.bar_style = 'danger'

    def _update_statistics(self):
        """Update statistics display."""
        if not self.vector_store:
            self.stats_html.value = '<p>No index built</p>'
            return

        stats = self.vector_store.get_statistics()

        self.stats_html.value = f"""
        <div style="padding: 15px; background: #e8f4e8; border-radius: 5px; margin-top: 10px;">
            <h4 style="margin-top: 0;">Index Statistics</h4>
            <table style="width: 100%;">
                <tr><td><b>Collection:</b></td><td>{stats['collection_name']}</td></tr>
                <tr><td><b>Total Vectors:</b></td><td>{stats['total_chunks']}</td></tr>
                <tr><td><b>Unique Sources:</b></td><td>{stats['unique_sources']}</td></tr>
                <tr><td><b>Embedding Dim:</b></td><td>{self.embedder.embedding_dim if self.embedder else 'N/A'}</td></tr>
                <tr><td><b>Model:</b></td><td>{self.model_dropdown.value}</td></tr>
            </table>
        </div>
        """

    def set_chunks(self, chunks: List[Chunk]):
        """Set chunks to embed."""
        self.chunks = chunks
        self.embedded_chunks = []

        with self.status_output:
            clear_output()
            print(f"Loaded {len(chunks)} chunks ready for embedding")

    def get_embedder(self) -> Optional[EmbeddingGenerator]:
        """Get the embedder."""
        return self.embedder

    def get_vector_store(self) -> Optional[VectorStore]:
        """Get the vector store."""
        return self.vector_store

    def get_embedded_chunks(self) -> List[EmbeddedChunk]:
        """Get embedded chunks."""
        return self.embedded_chunks

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return {
            'model': self.model_dropdown.value,
            'batch_size': self.batch_slider.value,
            'device': self.device_toggle.value,
            'collection_name': self.collection_input.value,
        }

    def display(self):
        """Display the widget."""
        title = widgets.HTML('<h3>Steps 3-4: Embedding & Vector Store</h3>')

        model_row = widgets.HBox([
            self.model_dropdown,
            self.device_toggle
        ])

        config_row = widgets.HBox([
            self.batch_slider,
            self.collection_input
        ])

        layout = widgets.VBox([
            title,
            widgets.HTML('<h4>Embedding Configuration:</h4>'),
            model_row,
            config_row,
            self.reset_checkbox,
            self.build_btn,
            self.progress_bar,
            self.status_output,
            self.stats_html,
        ])

        display(layout)
        return layout
