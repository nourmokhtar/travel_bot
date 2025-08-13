# retrival.py
import pandas as pd
import ast
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance, PayloadSchemaType
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from typing import List, Dict, Any, Optional
model = SentenceTransformer('all-MiniLM-L6-v2')
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.hNdrwMHDDLHEqkKkeiUBemv6ypz1Z1SXso8b3z9qKhs"
QDRANT_URL = "https://e80c833c-de7b-47d6-abcd-cf7fb67cbd18.us-east4-0.gcp.cloud.qdrant.io"

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

# ==============================
# 4. Recreate Collection with keyword indexing
# ==============================
collection_name = "travel_info"
# ==============================
# ==============================
# 6. Define query function
from typing import Optional, List, Dict
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

def query_qdrant(user_question: str, location: Optional[str] = None, top_k: int = 5) -> List[Dict[str, any]]:
    """Query Qdrant by city, country, or location_key."""
    q_embedding = model.encode([user_question])[0].tolist()

    query_filter = None
    if location:
        location = location.strip()
        if "-" in location:
            # location_key provided (country-city)
            query_filter = Filter(
                must=[FieldCondition(key="location_key", match=MatchValue(value=location))]
            )
        else:
            # Either city or country
            # First try city match
            query_filter = Filter(
                should=[
                    FieldCondition(key="city", match=MatchValue(value=location)),
                    FieldCondition(key="country", match=MatchValue(value=location))
                ],
                # At least one of them must match
                must=None
            )

    results = client.search(
        collection_name=collection_name,
        query_vector=q_embedding,
        limit=top_k,
        with_payload=True,
        query_filter=query_filter
    )

    return [
        {
            "location_key": r.payload.get("location_key"),
            "country": r.payload.get("country"),
            "city": r.payload.get("city"),
            "text": r.payload.get("text"),
            "score": r.score
        }
        for r in results
    ]


# ==============================
# 7. Test query
# ==============================
if __name__ == "__main__":
    question = "I want a 3-day itinerary in cairo exploring culture, dishes, and activities."
    results = query_qdrant(question, location="Tunis", top_k=8)

    if not results:
        print("No results found.")
    else:
        for idx, item in enumerate(results, start=1):
            print(f"Rank {idx}: {item['location_key']} (score: {item['score']})")
            print(item["text"][:500].replace("\n", " "), "...\n")
