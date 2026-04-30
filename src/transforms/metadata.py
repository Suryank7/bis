"""
Living Documentation Copilot - Metadata Enrichment

Enriches document chunks with metadata for tracking and source attribution.
All metadata is computed incrementally as documents flow through the pipeline.
"""

import pathway as pw
from datetime import datetime
import hashlib
import logging

logger = logging.getLogger(__name__)


@pw.udf
def generate_version_hash(chunk_text: str) -> str:
    """Generate a version hash for a chunk based on its content."""
    return hashlib.md5(chunk_text.encode()).hexdigest()[:8]


@pw.udf
def get_current_timestamp() -> str:
    """Get current ISO timestamp for indexing time."""
    return datetime.now().isoformat()


@pw.udf
def format_source_attribution(filename: str, chunk_index: int) -> str:
    """Create human-readable source attribution."""
    return f"{filename} (section {chunk_index + 1})"


def apply_metadata_enrichment(chunks_table: pw.Table) -> pw.Table:
    """
    Enrich chunks with additional metadata for tracking.
    
    Adds:
    - version_hash: Content-based version identifier
    - indexed_at: Timestamp when this version was indexed
    - source_attribution: Human-readable source reference
    
    Args:
        chunks_table: Table with chunk_text, chunk_id, document_id, etc.
        
    Returns:
        Enriched table with additional metadata columns
    """
    logger.info("🔄 Applying metadata enrichment...")
    
    enriched = chunks_table.select(
        # Keep all original columns
        chunk_text=chunks_table.chunk_text,
        chunk_id=chunks_table.chunk_id,
        chunk_index=chunks_table.chunk_index,
        document_id=chunks_table.document_id,
        source_path=chunks_table.source_path,
        filename=chunks_table.filename,
        # Add new metadata
        version_hash=generate_version_hash(chunks_table.chunk_text),
        indexed_at=get_current_timestamp(),
        source_attribution=format_source_attribution(
            chunks_table.filename,
            chunks_table.chunk_index
        ),
    )
    
    logger.info("✅ Metadata enrichment applied")
    
    return enriched


# Export public API
__all__ = [
    "generate_version_hash",
    "get_current_timestamp",
    "format_source_attribution",
    "apply_metadata_enrichment",
]
