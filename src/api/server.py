"""
Living Documentation Copilot - REST API Server

Exposes the RAG system via REST endpoints.
Uses a simple HTTP server that runs alongside the Pathway pipeline.

Endpoints:
- POST /v1/pw_ai_answer - Ask a question
- GET /v1/pw_list_documents - List indexed documents
- GET /health - Health check
- GET /stats - Pipeline statistics
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional, Callable, Dict, Any
import json
import logging
import threading
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger(__name__)


class RAGRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for RAG API endpoints."""
    
    # These are set by the server factory
    question_answerer = None
    get_stats_callback = None
    
    def log_message(self, format: str, *args):
        """Override to use logging module."""
        logger.info(f"[API] {args[0]} {args[1]} {args[2]}")
    
    def _set_headers(self, status: int = 200, content_type: str = "application/json"):
        """Set response headers."""
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
    
    def _send_json(self, data: Dict, status: int = 200):
        """Send JSON response."""
        self._set_headers(status, "application/json")
        response = json.dumps(data, indent=2)
        self.wfile.write(response.encode())
    
    def _send_error(self, message: str, status: int = 400):
        """Send error response."""
        self._send_json({"error": message}, status)
    
    def _read_body(self) -> Dict:
        """Read and parse JSON body."""
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            return {}
        
        body = self.rfile.read(content_length)
        try:
            return json.loads(body.decode())
        except json.JSONDecodeError:
            return {}
    
    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self._set_headers(200)
    
    def do_GET(self):
        """Handle GET requests."""
        path = urlparse(self.path).path
        
        if path == "/health":
            self._handle_health()
        elif path == "/stats":
            self._handle_stats()
        elif path == "/v1/pw_list_documents":
            self._handle_list_documents()
        elif path == "/":
            self._handle_root()
        else:
            self._send_error(f"Unknown endpoint: {path}", 404)
    
    def do_POST(self):
        """Handle POST requests."""
        path = urlparse(self.path).path
        
        if path == "/v1/pw_ai_answer":
            self._handle_question()
        elif path == "/query":
            # Alias for convenience
            self._handle_question()
        else:
            self._send_error(f"Unknown endpoint: {path}", 404)
    
    def _handle_root(self):
        """Root endpoint - API documentation."""
        docs = {
            "service": "Living Documentation Copilot",
            "version": "1.0.0",
            "description": "Real-time RAG system for live documentation",
            "endpoints": {
                "POST /v1/pw_ai_answer": "Ask a question (body: {prompt: string})",
                "POST /query": "Alias for /v1/pw_ai_answer",
                "GET /v1/pw_list_documents": "List indexed documents",
                "GET /health": "Health check",
                "GET /stats": "Pipeline statistics"
            }
        }
        self._send_json(docs)
    
    def _handle_health(self):
        """Health check endpoint."""
        health = {
            "status": "healthy",
            "service": "living-docs-copilot",
            "llm_available": self.question_answerer.llm_client.is_available() if self.question_answerer else False
        }
        self._send_json(health)
    
    def _handle_stats(self):
        """Pipeline statistics endpoint."""
        stats = {}
        if self.get_stats_callback:
            stats = self.get_stats_callback()
        else:
            stats = {
                "indexed_chunks": len(self.question_answerer._chunks_data) if self.question_answerer else 0,
                "llm_type": self.question_answerer.llm_client.client_type if self.question_answerer else "unknown"
            }
        self._send_json(stats)
    
    def _handle_list_documents(self):
        """List indexed documents."""
        if not self.question_answerer:
            self._send_json({"documents": []})
            return
        
        # Get unique documents from chunks
        seen = set()
        documents = []
        
        for chunk in self.question_answerer._chunks_data:
            doc_id = chunk.get("document_id", "")
            if doc_id and doc_id not in seen:
                seen.add(doc_id)
                documents.append({
                    "document_id": doc_id,
                    "filename": chunk.get("filename", ""),
                    "source_path": chunk.get("source_path", ""),
                })
        
        self._send_json({"documents": documents, "count": len(documents)})
    
    def _handle_question(self):
        """Handle question answering."""
        if not self.question_answerer:
            self._send_error("RAG system not initialized", 503)
            return
        
        body = self._read_body()
        
        # Support both 'prompt' and 'question' fields
        question = body.get("prompt") or body.get("question")
        
        if not question:
            self._send_error("Missing 'prompt' or 'question' in request body", 400)
            return
        
        try:
            result = self.question_answerer.answer(question)
            self._send_json(result)
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            self._send_error(f"Error: {str(e)}", 500)


class APIServer:
    """
    REST API Server for the Living Documentation Copilot.
    
    Runs in a background thread alongside the Pathway pipeline.
    """
    
    def __init__(
        self,
        question_answerer,
        host: str = "0.0.0.0",
        port: int = 8000,
        get_stats_callback: Optional[Callable] = None
    ):
        """
        Initialize the API server.
        
        Args:
            question_answerer: The RAG question answerer instance
            host: Host to bind to
            port: Port to listen on
            get_stats_callback: Optional callback to get pipeline stats
        """
        self.host = host
        self.port = port
        self.question_answerer = question_answerer
        self.get_stats_callback = get_stats_callback
        
        self._server = None
        self._thread = None
    
    def _create_handler_class(self):
        """Create a handler class with access to our instances."""
        qa = self.question_answerer
        stats_cb = self.get_stats_callback
        
        class ConfiguredHandler(RAGRequestHandler):
            question_answerer = qa
            get_stats_callback = stats_cb
        
        return ConfiguredHandler
    
    def start(self, blocking: bool = False):
        """
        Start the API server.
        
        Args:
            blocking: If True, block until server stops. If False, run in background.
        """
        handler_class = self._create_handler_class()
        self._server = HTTPServer((self.host, self.port), handler_class)
        
        logger.info("=" * 50)
        logger.info("🚀 API SERVER STARTED")
        logger.info(f"   URL: http://{self.host}:{self.port}")
        logger.info(f"   Question endpoint: POST /v1/pw_ai_answer")
        logger.info("=" * 50)
        
        if blocking:
            self._server.serve_forever()
        else:
            self._thread = threading.Thread(target=self._server.serve_forever)
            self._thread.daemon = True
            self._thread.start()
    
    def stop(self):
        """Stop the API server."""
        if self._server:
            self._server.shutdown()
            logger.info("🛑 API server stopped")
    
    def update_index(self, chunks: list):
        """Update the question answerer's index."""
        if self.question_answerer:
            self.question_answerer.update_index(chunks)


# Export public API
__all__ = [
    "APIServer",
    "RAGRequestHandler",
]
