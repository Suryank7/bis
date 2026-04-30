"""
Living Documentation Copilot - Main Application

This is the entry point that ties everything together:
1. Loads configuration
2. Creates file watcher for live document ingestion
3. Applies streaming transformations
4. Builds incremental vector index
5. Configures RAG pipeline
6. Starts API server
7. Runs continuously, updating as documents change

Run with: python -m src.main
Or:       python src/main.py
"""

import sys
import os
import time
import logging
from pathlib import Path
from typing import List, Dict
import threading

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import Pathway
import pathway as pw

# Import our modules
from src.config.settings import settings
from src.connectors.file_watcher import create_document_stream
from src.transforms.pipeline import create_transformation_pipeline
from src.indexing.embeddings import LocalEmbedder, compute_embedding
from src.rag.llm_client import LLMClient
from src.rag.question_answerer import RAGQuestionAnswerer
from src.api.server import APIServer


# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.logging.level),
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print startup banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║     🔮 LIVING DOCUMENTATION COPILOT                                   ║
║                                                                       ║
║     Real-time RAG system powered by Pathway                           ║
║     Answers evolve automatically as documents change                  ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


class LiveDocumentationCopilot:
    """
    The main Living Documentation Copilot application.
    
    This class orchestrates all components:
    - Document ingestion
    - Transformation pipeline
    - Vector indexing
    - RAG question answering
    - REST API
    
    The system runs continuously, watching the documents folder
    and updating the index in real-time.
    """
    
    def __init__(self):
        """Initialize all components."""
        self.settings = settings
        self.embedder = None
        self.llm_client = None
        self.question_answerer = None
        self.api_server = None
        self._chunks_cache = []
        self._update_lock = threading.Lock()
        
    def setup(self):
        """Set up all components."""
        logger.info("\n" + "=" * 60)
        logger.info("🔧 INITIALIZING COMPONENTS")
        logger.info("=" * 60)
        
        # Validate configuration
        if not settings.validate():
            raise RuntimeError("Configuration validation failed")
        
        settings.print_config()
        
        # Initialize embedder
        logger.info("\n📊 Loading embedding model...")
        self.embedder = LocalEmbedder(settings.embedding.model)
        logger.info(f"   Model: {settings.embedding.model}")
        logger.info(f"   Dimension: {self.embedder.dimension}")
        
        # Initialize LLM client
        logger.info("\n🤖 Initializing LLM client...")
        self.llm_client = LLMClient.create(
            provider=settings.llm.provider,
            ollama_base_url=settings.llm.ollama_base_url,
            ollama_model=settings.llm.ollama_model
        )
        
        # Initialize RAG question answerer
        logger.info("\n🧠 Setting up RAG pipeline...")
        self.question_answerer = RAGQuestionAnswerer(
            llm_client=self.llm_client,
            embedding_model=settings.embedding.model,
            top_k=settings.rag.top_k_results
        )
        
        # Initialize API server (but don't start yet)
        logger.info("\n🌐 Preparing API server...")
        self.api_server = APIServer(
            question_answerer=self.question_answerer,
            host=settings.server.host,
            port=settings.server.port,
            get_stats_callback=self._get_stats
        )
        
        logger.info("\n✅ All components initialized")
    
    def _get_stats(self) -> Dict:
        """Get pipeline statistics."""
        return {
            "indexed_chunks": len(self._chunks_cache),
            "llm_type": self.llm_client.client_type if self.llm_client else "unknown",
            "llm_available": self.llm_client.is_available() if self.llm_client else False,
            "embedding_model": settings.embedding.model,
            "docs_path": str(settings.documents.docs_path.absolute()),
        }
    
    def _process_chunks_table(self, key, row, time, is_addition):
        """Callback for processing chunk updates."""
        with self._update_lock:
            if is_addition:
                # Add or update chunk
                chunk_data = {
                    "chunk_text": row.chunk_text,
                    "chunk_id": row.chunk_id,
                    "chunk_index": row.chunk_index,
                    "document_id": row.document_id,
                    "source_path": row.source_path,
                    "filename": row.filename,
                    "embedding": row.embedding,
                }
                
                # Remove old version if exists
                self._chunks_cache = [
                    c for c in self._chunks_cache 
                    if c.get("chunk_id") != row.chunk_id
                ]
                
                # Add new version
                self._chunks_cache.append(chunk_data)
            else:
                # Remove chunk
                self._chunks_cache = [
                    c for c in self._chunks_cache 
                    if c.get("chunk_id") != row.chunk_id
                ]
            
            # Update question answerer
            if self.question_answerer:
                self.question_answerer.update_index(self._chunks_cache.copy())
                
            logger.info(f"{'➕' if is_addition else '➖'} Chunk update: {row.chunk_id[:50]}... (Total: {len(self._chunks_cache)})")
    
    def run_polling_mode(self):
        """
        Run in polling mode for simpler operation.
        
        This mode periodically scans the documents folder and updates
        the index. It's simpler than full streaming but still provides
        real-time updates.
        """
        logger.info("\n" + "=" * 60)
        logger.info("📡 STARTING IN POLLING MODE")
        logger.info("=" * 60)
        logger.info(f"   Watching: {settings.documents.docs_path.absolute()}")
        logger.info(f"   Supported: {settings.documents.supported_extensions}")
        logger.info("=" * 60 + "\n")
        
        # Start API server in background
        self.api_server.start(blocking=False)
        
        # Track file states
        last_file_states = {}
        
        try:
            while True:
                try:
                    # Scan documents folder
                    current_files = {}
                    docs_path = settings.documents.docs_path
                    
                    if docs_path.exists():
                        for ext in settings.documents.supported_extensions:
                            for file_path in docs_path.glob(f"*{ext}"):
                                try:
                                    stat = file_path.stat()
                                    current_files[str(file_path)] = (
                                        stat.st_mtime,
                                        stat.st_size
                                    )
                                except Exception:
                                    pass
                    
                    # Check for changes
                    changed = False
                    
                    # Detect additions and modifications
                    for path, (mtime, size) in current_files.items():
                        if path not in last_file_states:
                            logger.info(f"📄 New file detected: {Path(path).name}")
                            changed = True
                        elif last_file_states[path] != (mtime, size):
                            logger.info(f"📝 File modified: {Path(path).name}")
                            changed = True
                    
                    # Detect deletions
                    for path in last_file_states:
                        if path not in current_files:
                            logger.info(f"🗑️ File deleted: {Path(path).name}")
                            changed = True
                    
                    # Re-index if changes detected
                    if changed or not self._chunks_cache:
                        self._reindex_all(current_files)
                    
                    last_file_states = current_files
                    
                except Exception as e:
                    logger.error(f"Error in polling loop: {e}")
                
                # Wait before next poll
                time.sleep(2)
                
        except KeyboardInterrupt:
            logger.info("\n🛑 Shutting down...")
            self.api_server.stop()
    
    def _reindex_all(self, files: Dict):
        """Re-index all documents."""
        logger.info(f"🔄 Re-indexing {len(files)} documents...")
        
        new_chunks = []
        
        for file_path in files.keys():
            try:
                path = Path(file_path)
                
                # Read file content
                if path.suffix.lower() == ".pdf":
                    # Handle PDF
                    with open(path, "rb") as f:
                        content = f.read()
                    text = self._extract_pdf_text(content)
                else:
                    # Handle text files
                    with open(path, "r", encoding="utf-8", errors="replace") as f:
                        text = f.read()
                
                # Skip empty files
                if not text.strip():
                    continue
                
                # Chunk the text
                chunks = self._chunk_text(
                    text,
                    settings.rag.chunk_size,
                    settings.rag.chunk_overlap
                )
                
                # Create chunk records with embeddings
                for idx, chunk_text in enumerate(chunks):
                    if chunk_text.strip():
                        embedding = compute_embedding(chunk_text, settings.embedding.model)
                        
                        new_chunks.append({
                            "chunk_text": chunk_text,
                            "chunk_id": f"{file_path}::chunk_{idx}",
                            "chunk_index": idx,
                            "document_id": file_path,
                            "source_path": file_path,
                            "filename": path.name,
                            "embedding": embedding,
                            "source_attribution": f"{path.name} (section {idx + 1})"
                        })
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
        
        # Update cache and question answerer
        with self._update_lock:
            self._chunks_cache = new_chunks
            if self.question_answerer:
                self.question_answerer.update_index(new_chunks)
        
        logger.info(f"✅ Indexed {len(new_chunks)} chunks from {len(files)} documents")
    
    def _extract_pdf_text(self, data: bytes) -> str:
        """Extract text from PDF."""
        try:
            from io import BytesIO
            from pypdf import PdfReader
            
            reader = PdfReader(BytesIO(data))
            text_parts = []
            
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            
            return "\n\n".join(text_parts)
        except ImportError:
            return "[PDF support requires pypdf]"
        except Exception as e:
            return f"[PDF Error: {e}]"
    
    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into chunks."""
        import re
        
        if not text or not text.strip():
            return []
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            if end >= len(text):
                chunk = text[start:].strip()
                if chunk:
                    chunks.append(chunk)
                break
            
            # Try to break at sentence boundary
            chunk_text = text[start:end]
            last_period = max(
                chunk_text.rfind('. '),
                chunk_text.rfind('! '),
                chunk_text.rfind('? '),
            )
            
            if last_period > chunk_size // 2:
                end = start + last_period + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start <= 0:
                start = end
        
        return chunks
    
    def run(self):
        """Run the application."""
        print_banner()
        self.setup()
        self.run_polling_mode()


def main():
    """Main entry point."""
    app = LiveDocumentationCopilot()
    app.run()


if __name__ == "__main__":
    main()
