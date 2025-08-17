# search_agent_structured.py
import os
import asyncio
import requests
import uuid
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()

# Config from environment (same names as you already use)
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
DOCS_COLLECTION = os.getenv("QDRANT_DOCS_COLLECTION", "travel_info")

# Qdrant client + local embedding model for saving docs
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
st_model = SentenceTransformer("all-MiniLM-L6-v2")  # used to embed stored text

# LLM: prefer ask_together_llm then answer_question from your llm.py
try:
    from llm import ask_together_llm  # preferred (messages)
except Exception:
    ask_together_llm = None

try:
    from llm import answer_question   # fallback (string-based)
except Exception:
    answer_question = None

def _call_llm_for_structuring(system_msg: str, user_msg: str) -> str:
    """
    Synchronous LLM call that returns a plain-text structured block.
    We first try ask_together_llm (messages-style); otherwise fall back to answer_question.
    This function is synchronous; when used from async we call via asyncio.to_thread.
    """
    if ask_together_llm:
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
        return ask_together_llm(messages)
    if answer_question:
        # combine into a single prompt
        prompt = system_msg + "\n\n" + user_msg
        return answer_question(prompt)
    raise RuntimeError("No LLM function available — add ask_together_llm or answer_question in llm.py")

# ---------- Search & scraping ----------
def search_serper(query):
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    try:
        resp = requests.post(url, headers=headers, json={"q": query})
        resp.raise_for_status()
        results = resp.json().get("organic", [])
        return [r.get("link") for r in results if r.get("link")]
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

# ---------- Qdrant helpers ----------
def ensure_collection_exists():
    try:
        qdrant_client.get_collection(DOCS_COLLECTION)
    except Exception:
        qdrant_client.recreate_collection(
            collection_name=DOCS_COLLECTION,
            vectors_config={"size": 384, "distance": "Cosine"}
        )

def register_search_in_kb(question, answer_text, location_key=None, country=None, city=None):
    """
    Save plain-text 'answer_text' into Qdrant with payload fields that match your existing format.
    Embedding is created from the saved plain text using sentence-transformers.
    """
    try:
        payload = {
            "location_key": location_key or "unknown",
            "country": country or "unknown",
            "city": city or "unknown",
            "text": answer_text
        }
        vector = st_model.encode(answer_text).tolist()
        qdrant_client.upsert(
            collection_name=DOCS_COLLECTION,
            points=[{
                "id": str(uuid.uuid4()),
                "vector": vector,
                "payload": payload
            }]
        )
        print("✅ Registered structured plain-text in Qdrant:", payload["location_key"])
        return True
    except Exception as e:
        print("❌ Failed to register in KB:", e)
        return False

# ---------- LLM structuring (plain-text) ----------
async def structure_content_to_plaintext(raw_text: str, location_key: str = None, country: str = None, city: str = None) -> str:
    """
    Ask the LLM to produce the exact plain-text block format you want.
    Runs the synchronous LLM call in a thread (so this function is async-friendly).
    """
    system = (
        "You are a travel data assistant. Convert the raw scraped content into ONE single plain-text block "
        "matching this exact style (use '-' bullets, pipes '|' between fields as shown):\n\n"
        "Location: <location_key>\n"
        "Activities: - <Name> | <short description> | <target audience> | <duration> | <cost> | <tips>\n"
        "Restaurants: - <Name> | <cuisine> | <meals> | <signature dish> | <short description> | <price>\n"
        "Dishes: - <Name> | <notes/ingredients> | <when eaten> | <price>\n"
        "Accommodation: - <Name> | <type> | <price range> | <notes>\n"
        "Scams: - <scam type> | <description> | <how to avoid>\n"
        "Transport: - <destination or route> | <mode> | <company> | <frequency> | <duration> | <price range>\n"
        "Visa_Info: - <requirement> | <notes>\n\n"
        "Omit empty sections. Keep each line concise. Return ONLY the plain text (no JSON, no extra commentary)."
    )

    user_msg = f"Raw text to reformat (truncated to 60k chars):\n{raw_text[:60000]}\n\nLocation metadata: {location_key or ''} | {city or ''} | {country or ''}"

    structured = await asyncio.to_thread(_call_llm_for_structuring, system, user_msg)
    if not structured:
        # fallback to naive formatting if LLM fails
        return f"Location: {location_key or 'unknown'}\n\n{raw_text[:4000]}"

    return structured.strip()

# ---------- Main fallback pipeline ----------
async def search_agent_fallback(query: str, location_key: str = None, country: str = None, city: str = None) -> str:
    """
    Async fallback to:
      - Serper search
      - scrape top pages (3)
      - LLM structure to plain-text
      - save to Qdrant (payload.text contains the plain text)
      - return the plain text (for immediate use as context)
    """
    print(f"Fallback search agent activated for query: {query}")
    ensure_collection_exists()

    links = search_serper(query)
    if not links:
        return "Sorry, I couldn't find relevant information online."

    scraped_texts = []
    for link in links[:3]:
        t = await fetch_text(link)
        if t:
            scraped_texts.append(t)

    if not scraped_texts:
        return "Sorry, I couldn't retrieve detailed information from the web."

    combined = "\n\n".join(scraped_texts)

    # Structure via LLM into your plain-text format
    structured_plain = await structure_content_to_plaintext(combined, location_key=location_key, country=country, city=city)

    # Print the content that will be added to Qdrant (for debugging/confirmation)
    print("\n📦 Plain-text that will be saved to Qdrant (preview):")
    print("---------------------------------------------------")
    print(structured_plain[:4000])  # preview up to 4k chars
    print("---------------------------------------------------\n")

    # Save into Qdrant
    saved = register_search_in_kb(
        question=query,
        answer_text=structured_plain,
        location_key=location_key,
        country=country,
        city=city
    )
    if saved:
        print("REGISTEREDDDDDDDDDDDDDDD")
    return structured_plain
