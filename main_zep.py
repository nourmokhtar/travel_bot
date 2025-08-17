# main_with_zep.py

import os
import asyncio
import time
import json
from dotenv import load_dotenv

load_dotenv()

from query_handler import extract_location
from retrival import query_qdrant
from search_agent import search_agent_fallback, register_search_in_kb
from zep_python.client import Zep
from zep_python.types.message import Message

# ---------------- LLM Helper ----------------
try:
    from llm import ask_together_llm
    def call_llm(messages, user_query=None, location=None):
        return ask_together_llm(messages)
except Exception:
    from llm import answer_question
    def call_llm(messages, user_query=None, location=None):
        try:
            return answer_question(user_query, location=location)
        except Exception:
            return answer_question(messages)

# ---------------- Zep Setup ----------------
ZEP_API_KEY = os.getenv("ZEP_API_KEY")
ZEP_URL = os.getenv("ZEP_URL", "https://api.getzep.com")
if not ZEP_API_KEY:
    raise SystemExit("Set ZEP_API_KEY in your .env")

zep_client = Zep(api_key=ZEP_API_KEY, base_url=ZEP_URL)
SESSION_ID = os.getenv("ZEP_SESSION_ID", "travel_assistant_session")
USER_ID = os.getenv("ZEP_USER_ID", "default_user")

MAX_ZEP_LEN = 2400
session_facts_cache = {}  # instant recall

# ---------------- Zep Helpers ----------------
def ensure_session(session_id=SESSION_ID, user_id=USER_ID):
    try:
        return zep_client.memory.add_session(session_id=session_id, user_id=user_id, metadata={"app": "travel_bot"})
    except Exception:
        return zep_client.memory.get_session(session_id=session_id)

def add_message_to_session(session_id, role, content="", metadata=None):
    msg = Message(role=role, content=content)
    if metadata:
        try:
            setattr(msg, "metadata", metadata)
        except Exception:
            pass
    return zep_client.memory.add(session_id=session_id, messages=[msg])

def get_facts_from_zep(session_id):
    """Fetch facts stored in Zep messages metadata."""
    facts = {}
    try:
        session_messages = zep_client.memory.get_session_messages(session_id=session_id, limit=50)
        for msg in getattr(session_messages, "messages", []):
            metadata = getattr(msg, "metadata", {}) or {}
            prev_facts = metadata.get("facts")
            if isinstance(prev_facts, dict):
                for k, v in prev_facts.items():
                    if v and k not in facts:
                        facts[k] = v
    except Exception:
        pass
    return facts

# ---------------- Fact Extraction ----------------
# ---------------- Fact Extraction ----------------
import re

def extract_duration(user_message):
    """Fallback parser to extract duration in days if LLM misses it."""
    match = re.search(r"(\d+)\s*day", user_message, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None

def extract_relevant_facts(user_message):
    global session_facts_cache

    # Merge facts from Zep + in-memory cache
    merged_facts = {**get_facts_from_zep(SESSION_ID), **session_facts_cache}

    # Ask LLM to extract new facts with explicit JSON output
    prompt = [
        {"role": "system", "content":
         "You are a travel assistant. Extract all relevant travel info from the user message. "
         "Include: location, budget, duration (in days, numeric), preferences, activities, dates. "
         "Always output valid JSON with these keys. If a value is not mentioned, return null."},
        {"role": "user", "content": f"User message:\n{user_message}\nPreviously known facts:\n{json.dumps(merged_facts)}"}
    ]
    result = call_llm(prompt)

    try:
        new_facts = json.loads(result)
        if not isinstance(new_facts, dict):
            new_facts = {}
    except Exception:
        # fallback: just extract location
        location = extract_location(user_message)
        new_facts = {"location": location} if location else {}

    # Fallback extraction for duration if LLM missed it
    if ("duration" not in new_facts or not new_facts.get("duration")) and not merged_facts.get("duration"):
        duration = extract_duration(user_message)
        if duration:
            new_facts["duration"] = duration

    # Merge new facts into cache
    for k, v in new_facts.items():
        if v:
            merged_facts[k] = v

    session_facts_cache = dict(merged_facts)
    return merged_facts

# ---------------- Persistence Helpers ----------------
def summarize_text_sync(text, max_sentences=3):
    prompt = [
        {"role": "system", "content": "You are a concise summarizer."},
        {"role": "user", "content": f"Summarize the text below in {max_sentences} short sentences:\n{text}"}
    ]
    try:
        s = call_llm(prompt)
        return s if s else text[:1000] + "..."
    except Exception:
        return (text[:1000] + "...") if len(text) > 1000 else text

def clear_session_messages(session_id):
    try:
        resp = zep_client.memory.get_session_messages(session_id=session_id, limit=1000)
        messages = getattr(resp, "messages", []) or []
        for msg in messages:
            try:
                zep_client.memory.delete_message(message_id=msg.id)
            except Exception:
                pass
        print(f"🗑️ Cleared {len(messages)} messages from session '{session_id}'.")
    except Exception as e:
        print("⚠️ Failed to clear session messages:", e)

def persist_facts_and_answer(session_id, user_query, facts, answer):
    """Store both facts and assistant answer in Zep."""
    try:
        summary = summarize_text_sync(answer, max_sentences=3)
        if len(summary) > MAX_ZEP_LEN:
            summary = summary[:MAX_ZEP_LEN] + "..."
        add_message_to_session(session_id, role="user", content=user_query, metadata={"facts": facts})
        add_message_to_session(session_id, role="assistant", content=summary)
        print("✅ Facts & assistant answer stored in Zep.")
    except Exception as e:
        print("⚠️ Failed to store in Zep:", e)

# ---------------- Main Loop ----------------
async def main():
    clear_session_messages(SESSION_ID)
    ensure_session()
    print("✅ Zep session ready:", SESSION_ID)

    while True:
        user_query = input("📝 Enter your travel query (or 'exit'): ").strip()
        if user_query.lower() in ("exit", "quit"):
            break
        if not user_query:
            continue

        # Extract + store facts
        facts = extract_relevant_facts(user_query)
        print(f"📌 Facts for this session so far: {facts}")

        # Retrieve RAG context
        location = facts.get("location")
        context_results = query_qdrant(user_query, location=location, top_k=5)
        context_texts = "\n\n".join([r["text"] for r in context_results]) if context_results else ""

        if not context_results:
            print("🌐 No local data, using fallback search...")
            fallback_answer = await search_agent_fallback(user_query, location_key=location, country=location, city=None)
            context_texts = fallback_answer
            
        # Build LLM messages
         # 4) LLM messages (provide facts + readable context)
        messages = [
            {"role":"system", "content":
             "You are a world-class travel assistant. Use ONLY the provided context and the provided facts."},
            {"role":"user", "content":
             f"Facts so far:\n{json.dumps(facts, indent=2)}\n\nRAG context:\n{context_texts}\n\nUser Question:\n{user_query}\n\n"
             "Instructions: 1) Provide a day-by-day plan (morning, evening, night). "
             "2) Mention signature dishes & restaurants with approximate costs. "
             "3) Include accommodation suggestions with price ranges. "
             "4) Include transport options between locations with approximate fares. "
             "5) Provide safety tips & common scams. "
             "6) Add a final trip cost estimate. "
             "7) Include any unique local tips or must-see spots. "
             "8) Be concise, clear, and engaging."
            }
        ]
        start_time = time.time()
        try:
            answer = call_llm(messages, user_query=user_query, location=location)
        except Exception as e:
            print("❌ LLM call failed:", e)
            answer = "Sorry — I couldn't generate an answer right now."

        print(f"\n⏱️ LLM response took {time.time() - start_time:.2f}s\n")
        print("💡 Answer:\n", answer)

        # Store facts + answer in Zep
        persist_facts_and_answer(SESSION_ID, user_query, facts, answer)

if __name__ == "__main__":
    asyncio.run(main())