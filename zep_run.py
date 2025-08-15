# zep_test_run2.py
import os
from dotenv import load_dotenv
from pprint import pprint

load_dotenv()
API_KEY = os.getenv("ZEP_API_KEY")
ZEP_URL = os.getenv("ZEP_URL", "https://api.getzep.com")  # or http://localhost:8000 if local

if not API_KEY:
    raise SystemExit("Please set ZEP_API_KEY in your .env file")

# import the client class and Message DTO from the installed SDK
from zep_python.client import Zep
from zep_python.types.message import Message

# construct client (use base_url kwarg if needed)
try:
    client = Zep(api_key=API_KEY, base_url=ZEP_URL)
except TypeError:
    client = Zep(api_key=API_KEY, url=ZEP_URL)

print("Client constructed. client.memory methods:", [n for n in dir(client.memory) if not n.startswith("_")])

# 1) create a session (session_id and user_id required)
session_id = "test_session_1"
user_id = "user_test_1"
print(f"\nCreating session {session_id} for user {user_id} ...")
sess = client.memory.add_session(session_id=session_id, user_id=user_id, metadata={"source": "probe"})
print("Session created (server response):", sess)

# 2) add messages to session: must pass a sequence of Message objects
msg1 = Message(role="user", content="Hello I want to travel to France next month")
msg2 = Message(role="assistant", content="Great — how many days do you plan to stay?")
print(f"\nAdding {len([msg1,msg2])} messages to session {session_id} ...")
add_resp = client.memory.add(session_id=session_id, messages=[msg1, msg2])
print("Add response:", add_resp)

# 3) fetch session messages
print(f"\nFetching messages for session {session_id} ...")
msgs = client.memory.get_session_messages(session_id=session_id, limit=20)
print("Fetched messages (MessageListResponse):")
pprint(msgs)

# If you want to inspect the raw messages list inside the response object:
try:
    for m in msgs.messages:
        print(f" - {m.role}: {m.content}")
except Exception:
    pprint(msgs)
