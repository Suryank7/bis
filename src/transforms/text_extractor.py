"""
Living Documentation Copilot - Text Extraction

Extracts text content from binary file data based on file type.
Supports Markdown, TXT, and PDF formats.

All transformations are implemented as pure Pathway UDFs for
incremental streaming processing - no batch operations.
"""

import pathway as pw
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@pw.udf
def extract_text_from_binary(data: bytes, metadata: dict) -> str:
    """
    Extract text content from binary file data.
    
    This UDF is applied in a streaming fashion - it only processes
    changed documents, not the entire corpus.
    
    Args:
        data: Binary content of the file
        metadata: File metadata including path
        
    Returns:
        Extracted text content as string
    """
    path = metadata.get("path", "")
    extension = Path(path).suffix.lower()
    
    try:
        if extension in (".md", ".txt"):
            # Plain text files - decode directly
            return data.decode("utf-8", errors="replace")
        
        elif extension == ".pdf":
            # PDF files - use pypdf for extraction
            return _extract_pdf_text(data)
        
        else:
            # Unknown format - try as text
            return data.decode("utf-8", errors="replace")
            
    except Exception as e:
        logger.error(f"Error extracting text from {path}: {e}")
        return f"[Error extracting content from {Path(path).name}]"


def _extract_pdf_text(data: bytes) -> str:
    """
    Extract text from PDF binary data.
    
    Uses pypdf for lightweight PDF text extraction.
    Falls back to unstructured if pypdf fails.
    """
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
        return "[PDF support requires pypdf - pip install pypdf]"
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        return f"[Error reading PDF: {str(e)[:100]}]"


@pw.udf
def get_document_id(metadata: dict) -> str:
    """Extract unique document ID from metadata."""
    return metadata.get("path", "unknown")


@pw.udf
def get_source_path(metadata: dict) -> str:
    """Extract source file path from metadata."""
    return metadata.get("path", "unknown")


@pw.udf
def get_filename(metadata: dict) -> str:
    """Extract filename from metadata path."""
    path = metadata.get("path", "unknown")
    return Path(path).name


def apply_text_extraction(docs_table: pw.Table) -> pw.Table:
    """
    Apply text extraction transformation to a document table.
    
    Takes a table with binary data and metadata, returns a table with:
    - text: Extracted text content
    - document_id: Unique document identifier
    - source_path: Original file path
    - filename: Just the filename
    
    Args:
        docs_table: Table from file_watcher with data and _metadata columns
        
    Returns:
        Transformed table with text content and enriched metadata
    """
    logger.info("🔄 Applying text extraction transformation...")
    
    return docs_table.select(
        text=extract_text_from_binary(docs_table.data, docs_table._metadata),
        document_id=get_document_id(docs_table._metadata),
        source_path=get_source_path(docs_table._metadata),
        filename=get_filename(docs_table._metadata),
        _metadata=docs_table._metadata,  # Keep original metadata
    )


# Export public API
__all__ = [
    "extract_text_from_binary",
    "apply_text_extraction",
    "get_document_id",
    "get_source_path",
    "get_filename",
]
