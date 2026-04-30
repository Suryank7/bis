import re
import logging
from pathlib import Path
from pypdf import PdfReader

logger = logging.getLogger(__name__)

def load_bis_pdf(file_path: Path) -> str:
    """
    Extracts text from the BIS SP 21 PDF.
    Returns the raw text for the chunker to process.
    """
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return ""

    try:
        reader = PdfReader(str(file_path))
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        
        full_text = "\n\n".join(text_parts)
        logger.info(f"Successfully loaded {file_path.name} ({len(full_text)} chars)")
        return full_text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return ""
