import requests
import asyncio
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
import os
import uuid
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
from qdrant_client import QdrantClient
from langchain_qdrant import Qdrant

load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
DOCS_COLLECTION = os.getenv("QDRANT_DOCS_COLLECTION", "travel_info")


qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Ensure collection exists before saving
def ensure_collection_exists():
    try:
        qdrant_client.get_collection(DOCS_COLLECTION)
    except Exception:
        qdrant_client.recreate_collection(
            collection_name=DOCS_COLLECTION,
            vectors_config={"size": 384, "distance": "Cosine"}
        )

def search_serper(query):
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    try:
        response = requests.post(url, headers=headers, json={"q": query})
        response.raise_for_status()
        results = response.json().get('organic', [])
        return [item['link'] for item in results]
    except Exception as e:
        print("Search error:", e)
        return []

async def fetch_text(url):
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, timeout=60000)
            html = await page.content()
            await browser.close()
        return BeautifulSoup(html, "html.parser").get_text(separator="\n", strip=True)
    except Exception as e:
        print(f"Scraping error ({url}): {e}")
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser").get_text(separator="\n", strip=True)
        except Exception as e2:
            print(f"Fallback failed ({url}): {e2}")
            return ""

import uuid
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def register_search_in_kb(question, answer, location_key=None, country=None, city=None, source="search_agent_fallback"):
    try:
        vectorstore = Qdrant(
            client=qdrant_client,
            collection_name="travel_info",
            embeddings=embeddings
        )

        payload = {
            "location_key": location_key or "unknown",
            "country": country or "unknown",
            "city": city or "unknown",
            "text": f"Answer: {answer}"
        }

        # Generate embedding directly
        q_emb = model.encode(question).tolist()

        # Push to Qdrant manually with ID
        qdrant_client.upsert(
            collection_name="travel_info",
            points=[{
                "id": str(uuid.uuid4()),  # ✅ Required unique ID
                "vector": q_emb,
                "payload": payload
            }]
        )

        print("✅ Registered search result in travel_info collection.")
        return True

    except Exception as e:
        print(f"❌ Failed to register in KB: {e}")
        return False


async def search_agent_fallback(query: str) -> str:
    print(f"Fallback search agent activated for query: {query}")

    # Step 1: Search Google via Serper API
    links = search_serper(query)
    if not links:
        return "Sorry, I couldn't find relevant information online."

    # Step 2: Scrape content from top links (limit to 3)
    scraped_texts = []
    for link in links[:2]:
        text = await fetch_text(link)
        if text:
            scraped_texts.append(text)

    if not scraped_texts:
        return "Sorry, I couldn't retrieve detailed information from the web."

    # Step 3: Combine scraped texts as context for LLM prompt
    combined_context = "\n\n".join(scraped_texts)

    return combined_context
