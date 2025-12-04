"""
Query Widget - Phase B, Steps 5-8 UI

Interactive query interface for retrieval and generation.
"""
from typing import List, Dict, Any, Optional

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output, Markdown

from core.embedder import EmbeddingGenerator
from core.vector_store import VectorStore
from core.retriever import Retriever, RetrievalResult
from core.generator import ResponseGenerator, GenerationResult
from .visualization import create_similarity_chart


class QueryWidget:
    """
    Interactive query interface.

    Features:
    - Query input
    - Top-k and threshold controls
    - LLM configuration
    - Results display with similarity scores
    - Generated response display
    """

    def __init__(
        self,
        embedder: Optional[EmbeddingGenerator] = None,
        vector_store: Optional[VectorStore] = None
    ):
        """
        Initialize the query widget.

        Args:
            embedder: EmbeddingGenerator instance
            vector_store: VectorStore instance
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.retriever: Optional[Retriever] = None
        self.generator: Optional[ResponseGenerator] = None
        self.last_results: List[RetrievalResult] = []
        self.last_generation: Optional[GenerationResult] = None

        self._create_widgets()
        self._bind_events()

    def _create_widgets(self):
        """Create all UI components."""
        # Query input
        self.query_input = widgets.Textarea(
            placeholder='Enter your question here...',
            layout=widgets.Layout(width='100%', height='80px')
        )

        # Retrieval controls
        self.topk_slider = widgets.IntSlider(
            value=5,
            min=1,
            max=20,
            step=1,
            description='Top-K:',
            continuous_update=False,
            style={'description_width': '80px'},
            layout=widgets.Layout(width='250px')
        )

        self.threshold_slider = widgets.FloatSlider(
            value=0.0,
            min=0.0,
            max=1.0,
            step=0.05,
            description='Min Score:',
            continuous_update=False,
            style={'description_width': '80px'},
            layout=widgets.Layout(width='250px')
        )

        # LLM controls
        self.model_dropdown = widgets.Dropdown(
            options=[
                ('Gemini 2.5 Flash Lite', 'gemini-2.5-flash-lite'),
                ('Gemini 2.5 Flash', 'gemini-2.5-flash'),
                ('Gemini 1.5 Flash', 'gemini-1.5-flash'),
                ('Gemini 1.5 Pro', 'gemini-1.5-pro'),
            ],
            value='gemini-2.5-flash-lite',
            description='LLM Model:',
            style={'description_width': '80px'}
        )

        self.temp_slider = widgets.FloatSlider(
            value=0.7,
            min=0.0,
            max=2.0,
            step=0.1,
            description='Temperature:',
            continuous_update=False,
            style={'description_width': '100px'},
            layout=widgets.Layout(width='250px')
        )

        self.max_tokens_slider = widgets.IntSlider(
            value=1024,
            min=100,
            max=4000,
            step=100,
            description='Max Tokens:',
            continuous_update=False,
            style={'description_width': '100px'},
            layout=widgets.Layout(width='250px')
        )

        # System prompt
        self.system_prompt_text = widgets.Textarea(
            value=ResponseGenerator.DEFAULT_SYSTEM_PROMPT,
            description='System Prompt:',
            layout=widgets.Layout(width='100%', height='100px')
        )

        # Buttons
        self.search_btn = widgets.Button(
            description='Search',
            button_style='primary',
            icon='search'
        )

        self.generate_btn = widgets.Button(
            description='Generate Answer',
            button_style='success',
            icon='magic'
        )

        # Output areas
        self.retrieval_output = widgets.Output()
        self.generation_output = widgets.Output()
        self.status_output = widgets.Output()

    def _bind_events(self):
        """Bind widget events."""
        self.search_btn.on_click(self._on_search)
        self.generate_btn.on_click(self._on_generate)

    def _on_search(self, btn):
        """Handle search button click."""
        query = self.query_input.value.strip()
        if not query:
            with self.status_output:
                clear_output()
                print("Please enter a query")
            return

        if not self.embedder or not self.vector_store:
            with self.status_output:
                clear_output()
                print("Embedder and vector store must be set first")
            return

        with self.status_output:
            clear_output()
            print("Searching...")

        try:
            # Initialize retriever if needed
            if not self.retriever:
                self.retriever = Retriever(self.embedder, self.vector_store)

            # Perform retrieval
            self.last_results = self.retriever.retrieve(
                query=query,
                k=self.topk_slider.value,
                threshold=self.threshold_slider.value
            )

            with self.status_output:
                print(f"Found {len(self.last_results)} relevant chunks")

            # Display results
            self._display_retrieval_results()

        except Exception as e:
            with self.status_output:
                print(f"Error: {e}")

    def _on_generate(self, btn):
        """Handle generate button click."""
        query = self.query_input.value.strip()
        if not query:
            with self.status_output:
                clear_output()
                print("Please enter a query")
            return

        if not self.last_results:
            with self.status_output:
                clear_output()
                print("Please search first to retrieve context")
            return

        with self.status_output:
            clear_output()
            print("Generating response...")

        try:
            # Initialize generator if needed
            if not self.generator:
                self.generator = ResponseGenerator(
                    model=self.model_dropdown.value,
                    temperature=self.temp_slider.value,
                    max_tokens=self.max_tokens_slider.value
                )
            else:
                self.generator.update_parameters(
                    model=self.model_dropdown.value,
                    temperature=self.temp_slider.value,
                    max_tokens=self.max_tokens_slider.value
                )

            # Generate response
            self.last_generation = self.generator.generate(
                query=query,
                context_chunks=self.last_results,
                system_prompt=self.system_prompt_text.value
            )

            with self.status_output:
                print(f"Response generated ({self.last_generation.total_tokens} tokens)")

            # Display response
            self._display_generation_result()

        except Exception as e:
            with self.status_output:
                print(f"Error: {e}")

    def _display_retrieval_results(self):
        """Display retrieval results."""
        with self.retrieval_output:
            clear_output()

            if not self.last_results:
                print("No results found")
                return

            # Show similarity chart
            fig = create_similarity_chart(self.last_results)
            fig.show()

            # Show chunk details
            print("\n" + "=" * 60)
            print("RETRIEVED CHUNKS")
            print("=" * 60)

            for result in self.last_results:
                print(f"\n[Rank {result.rank}] Score: {result.score:.4f}")
                print(f"Source: {result.source}")
                print("-" * 40)
                print(result.text[:500] + "..." if len(result.text) > 500 else result.text)

    def _display_generation_result(self):
        """Display generation result."""
        with self.generation_output:
            clear_output()

            if not self.last_generation:
                print("No response generated")
                return

            # Response header
            display(HTML(f"""
            <div style="padding: 15px; background: #f0f8ff; border-radius: 5px; border-left: 4px solid #4a90d9;">
                <h4 style="margin-top: 0;">Generated Response</h4>
                <p style="color: #666; font-size: 0.9em;">
                    Model: {self.last_generation.model} |
                    Tokens: {self.last_generation.total_tokens} |
                    Sources: {', '.join(self.last_generation.sources)}
                </p>
            </div>
            """))

            # Response content
            display(Markdown(self.last_generation.response))

    def set_components(
        self,
        embedder: EmbeddingGenerator,
        vector_store: VectorStore
    ):
        """Set embedder and vector store."""
        self.embedder = embedder
        self.vector_store = vector_store
        self.retriever = Retriever(embedder, vector_store)

        with self.status_output:
            clear_output()
            print("Components initialized. Ready for queries!")

    def get_last_results(self) -> List[RetrievalResult]:
        """Get last retrieval results."""
        return self.last_results

    def get_last_generation(self) -> Optional[GenerationResult]:
        """Get last generation result."""
        return self.last_generation

    def display(self):
        """Display the widget."""
        title = widgets.HTML('<h3>Steps 5-8: Query & Generation</h3>')

        # Retrieval section
        retrieval_title = widgets.HTML('<h4>Retrieval Settings (Steps 5-7):</h4>')
        retrieval_controls = widgets.HBox([
            self.topk_slider,
            self.threshold_slider,
            self.search_btn
        ])

        # Generation section
        generation_title = widgets.HTML('<h4>Generation Settings (Step 8):</h4>')
        llm_controls = widgets.HBox([
            self.model_dropdown,
            self.temp_slider,
            self.max_tokens_slider
        ])

        layout = widgets.VBox([
            title,
            widgets.HTML('<h4>Query:</h4>'),
            self.query_input,
            retrieval_title,
            retrieval_controls,
            generation_title,
            llm_controls,
            self.system_prompt_text,
            self.generate_btn,
            self.status_output,
            widgets.HTML('<h4>Retrieved Chunks:</h4>'),
            self.retrieval_output,
            widgets.HTML('<h4>Generated Response:</h4>'),
            self.generation_output,
        ])

        display(layout)
        return layout
