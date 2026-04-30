"""
Living Documentation Copilot - Transformation Pipeline

Combines all transformation steps into a single pipeline:
1. Text Extraction (binary → text)
2. Chunking (text → chunks with overlap)
3. Metadata Enrichment (chunks → enriched chunks)

Usage:
    from src.transforms.pipeline import create_transformation_pipeline
    
    enriched_chunks = create_transformation_pipeline(
        docs_table,
        chunk_size=400,
        overlap=50
    )
"""

import pathway as pw
from typing import Optional
import logging

from .text_extractor import apply_text_extraction
from .chunker import apply_chunking
from .metadata import apply_metadata_enrichment

logger = logging.getLogger(__name__)


def create_transformation_pipeline(
    docs_table: pw.Table,
    chunk_size: int = 400,
    chunk_overlap: int = 50,
    skip_extraction: bool = False
) -> pw.Table:
    """
    Create the complete document transformation pipeline.
    
    This pipeline runs incrementally - when a document changes,
    only that document is re-processed, not the entire corpus.
    
    Pipeline Steps:
        Binary Data → Text Extraction → Chunking → Metadata Enrichment
    
    Args:
        docs_table: Table from file_watcher with data and _metadata columns
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between consecutive chunks
        skip_extraction: If True, assumes docs_table already has 'text' column
        
    Returns:
        Table with enriched chunks ready for embedding
        
    Output Schema:
        - chunk_text: str - The actual text content
        - chunk_id: str - Unique chunk identifier
        - chunk_index: int - Position in source document
        - document_id: str - Source document identifier
        - source_path: str - Original file path
        - filename: str - Original filename
        - version_hash: str - Content-based version
        - indexed_at: str - ISO timestamp
        - source_attribution: str - Human-readable source
    """
    logger.info("=" * 50)
    logger.info("🚀 INITIALIZING TRANSFORMATION PIPELINE")
    logger.info("=" * 50)
    
    # Step 1: Extract text from binary data
    if not skip_extraction:
        logger.info("\n📋 Step 1/3: Text Extraction")
        text_table = apply_text_extraction(docs_table)
    else:
        text_table = docs_table
    
    # Step 2: Split into overlapping chunks
    logger.info(f"\n📋 Step 2/3: Chunking (size={chunk_size}, overlap={chunk_overlap})")
    chunks_table = apply_chunking(text_table, chunk_size, chunk_overlap)
    
    # Step 3: Enrich with metadata
    logger.info("\n📋 Step 3/3: Metadata Enrichment")
    enriched_table = apply_metadata_enrichment(chunks_table)
    
    logger.info("\n✅ TRANSFORMATION PIPELINE READY")
    logger.info("   Pipeline will process documents incrementally as they change")
    logger.info("=" * 50)
    
    return enriched_table


# Export public API
__all__ = [
    "create_transformation_pipeline",
]
