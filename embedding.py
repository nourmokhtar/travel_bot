# retrival.py
import pandas as pd
import ast
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance, PayloadSchemaType
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from typing import List, Dict, Any, Optional

# ==============================
# 1. Load your CSV
# ==============================
df = pd.read_csv("travel_master.csv")

list_columns = ["accommodation", "activities", "dishes", "restaurants", "scams", "transport", "visa_info"]

# Convert string lists to Python lists
for col in list_columns:
    df[col] = df[col].apply(lambda x: ast.literal_eval(x) if pd.notna(x) and x != '' else [])

# Extract country and city from location_key
def extract_country_city(location_key):
    parts = location_key.split("-", 1)
    country = parts[0]
    city = parts[1] if len(parts) > 1 else None
    return country, city

df[['country', 'city']] = df['location_key'].apply(
    lambda x: pd.Series(extract_country_city(x))
)

# Combine text for embedding
def combine_text(row):
    parts = [f"Location: {row['location_key']}"]
    for col in list_columns:
        if row[col]:
            parts.append(f"{col.capitalize()}:")
            for item in row[col]:
                parts.append(f"- {item}")
    return "\n".join(parts)

df["combined_text"] = df.apply(combine_text, axis=1)

# ==============================
# 2. Create embeddings
# ==============================
print("🔄 Creating embeddings...")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df["combined_text"].tolist(), show_progress_bar=True)

# ==============================
# 3. Connect to Qdrant Cloud
# ==============================
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

# Delete existing collection (optional)
try:
    client.delete_collection(collection_name=collection_name)
    print(f"🗑️ Deleted existing collection '{collection_name}'.")
except Exception:
    pass
# Create collection without payload_schema
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=embeddings.shape[1],
        distance=Distance.COSINE
    )
)
print(f"✅ Collection '{collection_name}' created.")

# Create indexes for filtering (keyword index)
client.create_payload_index(
    collection_name=collection_name,
    field_name="location_key",
    field_schema=PayloadSchemaType.KEYWORD
)
client.create_payload_index(
    collection_name=collection_name,
    field_name="country",
    field_schema=PayloadSchemaType.KEYWORD
)
client.create_payload_index(
    collection_name=collection_name,
    field_name="city",
    field_schema=PayloadSchemaType.KEYWORD
)
print("✅ Payload indexes created for location_key, country, and city.")

# ==============================
# 5. Upload points in batches
# ==============================
print("📤 Uploading embeddings to Qdrant...")
points = [
    PointStruct(
        id=idx,
        vector=embedding.tolist(),
        payload={
            "location_key": row.location_key,
            "country": row.country,
            "city": row.city if pd.notna(row.city) else None,
            "text": row.combined_text
        }
    )
    for idx, (embedding, row) in enumerate(zip(embeddings, df.itertuples()))
]

batch_size = 50
for i in range(0, len(points), batch_size):
    client.upsert(
        collection_name=collection_name,
        points=points[i: i + batch_size]
    )

print("✅ All embeddings uploaded successfully!")
