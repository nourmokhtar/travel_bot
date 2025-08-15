import os
import json
from dotenv import load_dotenv
from zep_python.client import Zep
from zep_python.types.message import Message

load_dotenv()

ZEP_API_KEY = os.getenv("ZEP_API_KEY")
ZEP_URL = os.getenv("ZEP_URL", "https://api.getzep.com")
SESSION_ID = os.getenv("ZEP_SESSION_ID", "travel_assistant_session")

if not ZEP_API_KEY:
    raise SystemExit("Set ZEP_API_KEY in your .env")

zep_client = Zep(api_key=ZEP_API_KEY, base_url=ZEP_URL)

def view_zep_session(session_id=SESSION_ID, max_messages=100):
    """Print all messages and stored facts from the Zep session."""
    session_messages = zep_client.memory.get_session_messages(session_id=session_id, limit=max_messages)
    messages = getattr(session_messages, "messages", []) or []

    print(f"🗂️ Total messages in session '{session_id}': {len(messages)}\n")

    for i, msg in enumerate(messages, start=1):
        role = getattr(msg, "role", "unknown")
        content = getattr(msg, "content", "")
        metadata = getattr(msg, "metadata", {})
        facts = metadata.get("facts") if metadata else None

        print(f"--- Message {i} ---")
        print(f"Role: {role}")
        print(f"Content: {content}")
        if facts:
            print(f"Facts stored: {facts}")
        print("\n")

# Call it
view_zep_session()
