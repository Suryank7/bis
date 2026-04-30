"""
Living Documentation Copilot - File Watcher Connector

This module implements a streaming file watcher that monitors a directory
for document changes (additions, modifications, deletions) and streams
them into the Pathway pipeline for real-time processing.

Key Features:
- Monitors directory for .md, .txt, .pdf files
- Streams file content with metadata
- Automatically detects add/modify/delete events
- Works in streaming mode for real-time updates
"""

import pathway as pw
from pathlib import Path
from typing import List
import logging

logger = logging.getLogger(__name__)


class FileInputSchema(pw.Schema):
    """
    Schema for ingested file data.
    
    Pathway automatically tracks:
    - data: Raw binary content of the file
    - _metadata: File metadata including path, size, modified time
    """
    data: bytes


def create_document_stream(
    docs_path: Path,
    extensions: tuple = (".md", ".txt", ".pdf"),
    mode: str = "streaming"
) -> pw.Table:
    """
    Create a streaming table that watches a directory for document changes.
    
    This is the entry point for all documents into the RAG pipeline.
    The table automatically updates when files are added, modified, or deleted.
    
    Args:
        docs_path: Path to the directory to watch
        extensions: Tuple of file extensions to include
        mode: "streaming" for live updates, "static" for one-time read
    
    Returns:
        pw.Table: A Pathway table that streams file contents with metadata
    
    Example:
        >>> docs = create_document_stream(Path("./data/docs"))
        >>> # Table automatically updates as files change
    """
    # Ensure path exists
    docs_path = Path(docs_path)
    docs_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📂 Initializing file watcher on: {docs_path.absolute()}")
    logger.info(f"   Watching for extensions: {extensions}")
    logger.info(f"   Mode: {mode}")
    
    # Create file pattern filter
    # We'll filter by extension after reading since fs.read uses glob patterns
    
    # Read all files from directory in streaming mode
    # This connector automatically:
    # - Detects new files → adds rows
    # - Detects modified files → updates rows
    # - Detects deleted files → removes rows
    docs_table = pw.io.fs.read(
        path=str(docs_path),
        format="binary",
        mode=mode,
        with_metadata=True,  # Include file path, size, modified time
    )
    
    # Filter by extension - only keep supported document types
    docs_table = docs_table.filter(
        pw.apply(
            lambda meta: any(
                meta.get("path", "").lower().endswith(ext) 
                for ext in extensions
            ),
            docs_table._metadata
        )
    )
    
    logger.info("✅ File watcher initialized successfully")
    
    return docs_table


def get_file_info_from_metadata(metadata: dict) -> dict:
    """
    Extract useful file information from Pathway metadata.
    
    Args:
        metadata: The _metadata dict from pw.io.fs.read
        
    Returns:
        dict with document_id, path, filename, extension, modified_at
    """
    path = metadata.get("path", "unknown")
    path_obj = Path(path)
    
    return {
        "document_id": str(path),  # Use path as unique ID
        "path": str(path),
        "filename": path_obj.name,
        "extension": path_obj.suffix.lower(),
        "modified_at": metadata.get("modified_at", None),
        "size_bytes": metadata.get("size", 0),
    }


# Export public API
__all__ = [
    "FileInputSchema",
    "create_document_stream",
    "get_file_info_from_metadata",
]
