import re
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

def standard_aware_chunking(text: str, chunk_size: int = 500, overlap: int = 100) -> List[Dict]:
    """
    Splits text into chunks, but attempts to extract the IS Standard Number
    and Title to prefix every chunk. This ensures the embedding model always
    knows WHICH standard the chunk belongs to, maximizing Hit Rate.
    """
    if not text.strip():
        return []

    chunks = []
    
    # A heuristic regex for Indian Standards: e.g., "IS 269 : 2015 Ordinary Portland Cement"
    # This may need adjustment based on the exact PDF layout.
    is_pattern = re.compile(r'(IS\s+\d+[\w\s:-]+)', re.IGNORECASE)
    
    # Split text roughly by paragraphs or large sections to find headers
    sections = re.split(r'\n\n+', text)
    
    current_standard_context = "Unknown Standard"
    
    for section in sections:
        section = section.strip()
        if not section:
            continue
            
        # Check if this section contains a standard header
        match = is_pattern.search(section)
        if match:
            # We assume the match (often a title line) gives us context
            # We might want to cap the length of the context
            context_str = match.group(1).strip()
            if len(context_str) < 200: 
                current_standard_context = context_str
                
        # Now chunk the section using sliding window
        start = 0
        while start < len(section):
            end = start + chunk_size
            
            # Find a good break point (period)
            if end < len(section):
                last_period = section.rfind('.', start, end)
                if last_period != -1 and last_period > start + (chunk_size // 2):
                    end = last_period + 1
                    
            chunk_text = section[start:end].strip()
            if chunk_text:
                # [INNOVATION] Inject the standard context into the chunk!
                enriched_text = f"Standard: {current_standard_context}\nContent: {chunk_text}"
                chunks.append({
                    "chunk_id": f"chunk_{len(chunks)}",
                    "text": enriched_text,
                    "metadata": {
                        "standard": current_standard_context
                    }
                })
                
            start = end - overlap
            
    logger.info(f"Generated {len(chunks)} standard-aware chunks.")
    return chunks
