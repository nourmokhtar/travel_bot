# main_with_zep.py
import os
import sys
import re
import json
import time
import asyncio
import subprocess
from dotenv import load_dotenv

# ---------------- Load Environment ----------------
load_dotenv()
ZEP_API_KEY = os.getenv("ZEP_API_KEY")
ZEP_URL = os.getenv("ZEP_URL", "https://api.getzep.com")
SESSION_ID = os.getenv("ZEP_SESSION_ID", "travel_assistant_session")
USER_ID = os.getenv("ZEP_USER_ID", "default_user")

if not ZEP_API_KEY:
    raise SystemExit("⚠️ Set ZEP_API_KEY in your .env")

# ---------------- Custom Modules ----------------
from query_handler import extract_location
from retrival import query_qdrant
from search_agent import search_agent_fallback
from zep_python.client import Zep
from zep_python.types.message import Message

# Attempt to import different LLM handlers
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

# ---------------- Zep Client Setup ----------------
zep_client = Zep(api_key=ZEP_API_KEY, base_url=ZEP_URL)
MAX_ZEP_LEN = 2400
session_facts_cache = {}  # instant recall

# ---------------- Zep Helpers ----------------
def ensure_session(session_id=SESSION_ID, user_id=USER_ID):
    """Create or fetch a Zep session."""
    try:
        return zep_client.memory.add_session(
            session_id=session_id, user_id=user_id, metadata={"app": "travel_bot"}
        )
    except Exception:
        return zep_client.memory.get_session(session_id=session_id)

def add_message_to_session(session_id, role, content="", metadata=None):
    """Add a message to Zep session."""
    msg = Message(role=role, content=content)
    if metadata:
        setattr(msg, "metadata", metadata)
    return zep_client.memory.add(session_id=session_id, messages=[msg])

def get_facts_from_zep(session_id):
    """Retrieve facts stored in Zep session messages metadata."""
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

def clear_session_messages(session_id):
    """Delete all messages from a Zep session."""
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

# ---------------- Fact Extraction ----------------
def extract_duration(user_message):
    """Fallback parser to extract trip duration in days."""
    match = re.search(r"(\d+)\s*day", user_message, re.IGNORECASE)
    return int(match.group(1)) if match else None

def extract_relevant_facts(user_message):
    """Extract and merge travel facts from user input and Zep cache."""
    global session_facts_cache
    merged_facts = {**get_facts_from_zep(SESSION_ID), **session_facts_cache}

    prompt = [
        {"role": "system", "content":
         "You are a travel assistant. Extract all relevant travel info from the user message. "
         "Include: location, budget, duration (numeric days), preferences, activities, dates. "
         "Always output valid JSON with these keys. Return null if missing."},
        {"role": "user", "content":
         f"User message:\n{user_message}\nPreviously known facts:\n{json.dumps(merged_facts)}"}
    ]
    result = call_llm(prompt)

    try:
        new_facts = json.loads(result)
        if not isinstance(new_facts, dict):
            new_facts = {}
    except Exception:
        location = extract_location(user_message)
        new_facts = {"location": location} if location else {}

    # Fallback duration extraction
    if ("duration" not in new_facts or not new_facts.get("duration")) and not merged_facts.get("duration"):
        duration = extract_duration(user_message)
        if duration:
            new_facts["duration"] = duration

    # Merge into cache
    for k, v in new_facts.items():
        if v:
            merged_facts[k] = v
    session_facts_cache = dict(merged_facts)
    return merged_facts

async def ask_for_missing_facts(facts):
    """Prompt user to fill missing facts interactively."""
    # Location
    if not facts.get("location"):
        loc = input("🔹 What is your trip destination? (City or Country): ").strip()
        facts["location"] = loc.capitalize() if loc else None

    # Duration
    if not facts.get("duration"):
        while True:
            dur = input("🔹 How many days is your trip? Enter a number: ").strip()
            if dur.isdigit() and int(dur) > 0:
                facts["duration"] = int(dur)
                break
            print("⚠️ Enter a valid positive number.")

    # Budget
    if not facts.get("budget"):
        print("🔹 Budget:\n1) <$500\n2) $500-$1000\n3) >$1000")
        choice = input("Select option (1-3): ").strip()
        facts["budget"] = {"1": 500, "2": 1000, "3": 2000}.get(choice, 500)

    # Preferences
    if not facts.get("preferences"):
        print("🔹 Experience type:\n1) Adventure\n2) Relaxation\n3) Culture\n4) Food\n5) Shopping")
        choice = input("Select option (1-5): ").strip()
        prefs_map = {"1": "Adventure", "2": "Relaxation", "3": "Culture", "4": "Food", "5": "Shopping"}
        facts["preferences"] = prefs_map.get(choice, "Adventure")
    return facts

# ---------------- Persistence ----------------
def summarize_text_sync(text, max_sentences=3):
    """Summarize long text for Zep storage."""
    prompt = [
        {"role": "system", "content": "You are a concise summarizer."},
        {"role": "user", "content": f"Summarize the text below in {max_sentences} short sentences:\n{text}"}
    ]
    try:
        return call_llm(prompt) or text[:1000] + "..."
    except Exception:
        return text[:1000] + "..." if len(text) > 1000 else text

def persist_facts_and_answer(session_id, user_query, facts, answer):
    """Store facts + assistant answer in Zep session."""
    try:
        summary = summarize_text_sync(answer, max_sentences=3)
        if len(summary) > MAX_ZEP_LEN:
            summary = summary[:MAX_ZEP_LEN] + "..."
        add_message_to_session(session_id, "user", user_query, {"facts": facts})
        add_message_to_session(session_id, "assistant", summary)
        print("✅ Facts & assistant answer stored in Zep.")
    except Exception as e:
        print("⚠️ Failed to store in Zep:", e)

# ---------------- Main Loop ----------------
async def main():
    clear_session_messages(SESSION_ID)
    ensure_session()
    print("🛫 Welcome to BASSIR! Your personalized travel assistant.\n")
    print("💡 Type 'custom' or 'personalized' for a personalized trip plan.")
    print("💡 Type 'recap' anytime to see a summary.\n")

    while True:
        user_query = input("📝 Enter your travel query (or 'exit'): ").strip()
        if user_query.lower() in ("exit", "quit"):
            print("👋 Goodbye! Have a great trip!")
            break
        if not user_query:
            print("⚠️ Please enter a query.")
            continue

        # Recap
        if user_query.lower() == "recap":
            print("📄 Showing conversation recap...")
            try:
                subprocess.run([sys.executable, "zep_view.py"], check=True)
            except Exception as e:
                print("⚠️ Failed to execute zep_view.py:", e)
            continue

        # Determine personalized mode
        personalized_mode = any(word in user_query.lower() for word in ["personalized", "custom"])

        # Extract & confirm facts
        facts = extract_relevant_facts(user_query)
        if personalized_mode:
            facts = await ask_for_missing_facts(facts)
            print("\n📌 Let's confirm the details:")
            print(f"Destination: {facts.get('location')}")
            print(f"Duration: {facts.get('duration')} days")
            print(f"Budget: ${facts.get('budget')}")
            print(f"Experience type: {facts.get('preferences')}")
            if input("Are these correct? (yes/no): ").strip().lower() not in ("yes", "y"):
                facts = await ask_for_missing_facts(facts)

        # Retrieve context
        print("\n🌐 Fetching travel data...")
        rag_input_text = "Plan a personalized trip with these facts:\n" + json.dumps(facts, indent=2)
        location = facts.get("location")
        context_results = query_qdrant(rag_input_text, location=location, top_k=5)
        context_texts = "\n\n".join([r["text"] for r in context_results]) if context_results else ""

        # Fallback search
        if not context_results:
            print("🌐 No local data, using fallback search...")
            context_texts = await search_agent_fallback(user_query, location_key=location, country=location, city=None)

        # Build LLM messages
        messages = [
            {"role": "system", "content":
             "You are a world-class travel assistant. Use ONLY the provided context and the provided facts."},
            {"role": "user", "content":
             f"Facts so far:\n{json.dumps(facts, indent=2)}\n\nRAG context:\n{context_texts}\n\nUser Question:\n{llm_user_query}\n\n"
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
        # Generate answer
        start_time = time.time()
        try:
            answer = call_llm(messages, user_query=user_query, location=location)
        except Exception as e:
            print("❌ LLM call failed:", e)
            answer = "Sorry — I couldn't generate an answer right now."
        print(f"\n⏱️ LLM response took {time.time() - start_time:.2f}s\n")
        print("💡 Your Trip Plan:\n", answer)

        # Persist
        persist_facts_and_answer(SESSION_ID, user_query, facts, answer)
        print("\n--------------------------------------------\n")


if __name__ == "__main__":
    asyncio.run(main())
