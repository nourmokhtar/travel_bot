# llm.py
import os
from dotenv import load_dotenv
from together import Together
from retrival import query_qdrant  # your existing retrieval module

load_dotenv()

# ==============================
# Together API setup
# ==============================
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
client = Together(api_key=TOGETHER_API_KEY)
MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"

def ask_together_llm(messages):
    """
    messages: list of dicts with keys: 'role' and 'content'
    returns: string answer from LLM
    """
    formatted = [{"role": m["role"], "content": m["content"]} for m in messages]
    response = client.chat.completions.create(
        model=MODEL,
        messages=formatted,
        stream=False,
    )
    return response.choices[0].message.content.strip()


# ==============================
# RAG pipeline
# ==============================
def answer_question(question: str, location: str = None, top_k: int = 5) -> str:
    """
    Retrieve top-k documents from Qdrant, then ask LLaMA to answer using that context.
    """
    docs = query_qdrant(question, location=location, top_k=top_k)
   
    if not docs:
        return "No relevant documents found in the database."

    # Combine text from retrieved docs
    context = "\n\n".join([doc['text'] for doc in docs])


    messages = [
        {"role": "system", "content": "You are a helpful travel assistant."},
        {"role": "user", "content": f"Use the context below to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}"}
    ]

    return ask_together_llm(messages)


# ==============================
# Example usage
# ==============================
if __name__ == "__main__":
    question = "I want a 3-day itinerary in Tunis exploring culture, dishes, and activities."
    location = "Tunis"
    answer = answer_question(question, location=location, top_k=5)
    print("📝 Answer from LLaMA 3 8B:")
    print(answer)
