"""
Upload Widget - Phase A, Step 1 UI

Interactive file upload widget for document ingestion.
"""
from typing import List, Dict, Any, Callable, Optional
from pathlib import Path

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

from core.document_loader import DocumentLoader, Document


class UploadWidget:
    """
    Interactive file upload widget.

    Features:
    - Drag-and-drop file upload
    - Multiple file selection
    - File type validation
    - File list display with metadata
    """

    def __init__(self, on_documents_loaded: Optional[Callable[[List[Document]], None]] = None):
        """
        Initialize the upload widget.

        Args:
            on_documents_loaded: Callback when documents are loaded
        """
        self.documents: List[Document] = []
        self.on_documents_loaded = on_documents_loaded

        # Create widgets
        self._create_widgets()
        self._bind_events()

    def _create_widgets(self):
        """Create all UI components."""
        # File upload widget
        self.file_upload = widgets.FileUpload(
            accept='.pdf,.txt,.docx',
            multiple=True,
            description='Upload Files',
            button_style='primary',
            layout=widgets.Layout(width='200px')
        )

        # Load from path input
        self.path_input = widgets.Text(
            placeholder='Or enter file/directory path...',
            description='Path:',
            layout=widgets.Layout(width='400px')
        )

        self.load_path_btn = widgets.Button(
            description='Load Path',
            button_style='info',
            icon='folder-open'
        )

        # Clear button
        self.clear_btn = widgets.Button(
            description='Clear All',
            button_style='danger',
            icon='trash'
        )

        # Status output
        self.status_output = widgets.Output()

        # File list display
        self.file_list_output = widgets.Output()

        # Statistics display
        self.stats_html = widgets.HTML(value='<p>No documents loaded</p>')

    def _bind_events(self):
        """Bind widget events."""
        self.file_upload.observe(self._on_upload, names='value')
        self.load_path_btn.on_click(self._on_load_path)
        self.clear_btn.on_click(self._on_clear)

    def _on_upload(self, change):
        """Handle file upload."""
        if not change['new']:
            return

        with self.status_output:
            clear_output()
            print("Processing uploaded files...")

        new_docs = []
        for filename, file_info in change['new'].items():
            try:
                content = file_info['content']
                doc = DocumentLoader.load_from_bytes(content, filename)
                new_docs.append(doc)
                with self.status_output:
                    print(f"  Loaded: {filename} ({len(doc)} chars)")
            except Exception as e:
                with self.status_output:
                    print(f"  Error loading {filename}: {e}")

        self.documents.extend(new_docs)
        self._update_display()

        if self.on_documents_loaded and new_docs:
            self.on_documents_loaded(self.documents)

    def _on_load_path(self, btn):
        """Handle loading from path."""
        path = self.path_input.value.strip()
        if not path:
            with self.status_output:
                clear_output()
                print("Please enter a path")
            return

        with self.status_output:
            clear_output()
            print(f"Loading from: {path}")

        try:
            path_obj = Path(path)
            if path_obj.is_file():
                doc = DocumentLoader.load(path)
                self.documents.append(doc)
                with self.status_output:
                    print(f"  Loaded: {doc.source} ({len(doc)} chars)")
            elif path_obj.is_dir():
                docs = DocumentLoader.load_directory(path)
                self.documents.extend(docs)
                with self.status_output:
                    print(f"  Loaded {len(docs)} documents from directory")
            else:
                with self.status_output:
                    print(f"  Path not found: {path}")
                return

            self._update_display()

            if self.on_documents_loaded:
                self.on_documents_loaded(self.documents)

        except Exception as e:
            with self.status_output:
                print(f"  Error: {e}")

    def _on_clear(self, btn):
        """Clear all documents."""
        self.documents = []
        self._update_display()

        with self.status_output:
            clear_output()
            print("All documents cleared")

        if self.on_documents_loaded:
            self.on_documents_loaded(self.documents)

    def _update_display(self):
        """Update the file list and statistics display."""
        with self.file_list_output:
            clear_output()
            if not self.documents:
                display(HTML('<p><i>No documents loaded</i></p>'))
            else:
                # Create table
                rows = []
                for i, doc in enumerate(self.documents):
                    rows.append(f"""
                        <tr>
                            <td>{i + 1}</td>
                            <td>{doc.source}</td>
                            <td>{doc.file_type.upper()}</td>
                            <td>{len(doc):,}</td>
                            <td>{doc.metadata.get('page_count', '-')}</td>
                        </tr>
                    """)

                table_html = f"""
                <table style="width:100%; border-collapse: collapse;">
                    <thead>
                        <tr style="background-color: #f0f0f0;">
                            <th style="padding: 8px; border: 1px solid #ddd;">#</th>
                            <th style="padding: 8px; border: 1px solid #ddd;">Filename</th>
                            <th style="padding: 8px; border: 1px solid #ddd;">Type</th>
                            <th style="padding: 8px; border: 1px solid #ddd;">Characters</th>
                            <th style="padding: 8px; border: 1px solid #ddd;">Pages</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(rows)}
                    </tbody>
                </table>
                """
                display(HTML(table_html))

        # Update statistics
        if self.documents:
            total_chars = sum(len(d) for d in self.documents)
            self.stats_html.value = f"""
            <div style="padding: 10px; background: #e8f4e8; border-radius: 5px; margin-top: 10px;">
                <b>Summary:</b> {len(self.documents)} document(s), {total_chars:,} total characters
            </div>
            """
        else:
            self.stats_html.value = '<p>No documents loaded</p>'

    def add_document(self, doc: Document):
        """Add a document programmatically."""
        self.documents.append(doc)
        self._update_display()

    def get_documents(self) -> List[Document]:
        """Get all loaded documents."""
        return self.documents

    def display(self):
        """Display the widget."""
        title = widgets.HTML('<h3>Step 1: Document Upload</h3>')

        upload_row = widgets.HBox([
            self.file_upload,
            self.path_input,
            self.load_path_btn,
            self.clear_btn
        ])

        layout = widgets.VBox([
            title,
            upload_row,
            widgets.HTML('<h4>Loaded Documents:</h4>'),
            self.file_list_output,
            self.stats_html,
            self.status_output
        ])

        display(layout)
        self._update_display()

        return layout
