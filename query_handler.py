import re
from typing import Optional, Tuple
from retrival import query_qdrant  # your RAG retriever
def extract_location(query: str) -> Optional[str]:
    """
    Extract a country, city, or location_key from the user query.
    Improved: only capture the first word after 'in' or 'at'.
    """
    query_lower = query.lower()
    
    # Stop at end of sentence, comma, or 'exploring', 'visiting', etc.
    match = re.search(r'\b(?:in|at)\s+([a-zA-Z\-]+)', query_lower)
    if match:
        loc = match.group(1).strip()
        loc = loc.capitalize()  # match your DB casing
        return loc
    
    return None

