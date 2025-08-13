from query_handler import extract_location
from retrival import query_qdrant
from llm import answer_question
import time  # ← import the time module

# ==============================
# Main flow
# ==============================
def main():
    user_query = input("📝 Enter your travel query: ")
    
    location = extract_location(user_query)
    if location:
        print(f"🔍 Extracted location: {location}")
    else:
        print("⚠️ No location detected, querying globally...")

    # ==============================
    # Retrieve context from Qdrant
    # ==============================
    start_time = time.time()
    context_results = query_qdrant(user_query, location=location, top_k=5)
    retrieval_time = time.time() - start_time
    print(f"⏱️ Context retrieval took {retrieval_time:.2f} seconds\n")

    context_texts = "\n\n".join([r["text"] for r in context_results])

    # ==============================
    # Prepare messages for LLM
    # ==============================
    messages = [
        {
            "role": "system",
            "content": (
                "You are a world-class travel assistant. "
                "You provide detailed, friendly, and practical travel advice. "
                "Use the provided context to create personalized travel itineraries, "
                "recommend local activities, accommodations, restaurants, dishes, transport options, "
                "and provide estimated price ranges. "
                "dont give anything from your knowledge only from the context . "
                "tips to be safe and avoid scams  "
                "Highlight must-see attractions."
                "at the very end add an estimation for the trip . "
                "If context is missing,  say you dont know ."
            )
        },
        {
            "role": "user",
            "content": (
                f"Context (from travel database):\n{context_texts}\n\n"
                f"User Question:\n{user_query}\n\n"
                "Instructions: "
                "1. Suggest a complete itinerary with day-by-day plan. morning , evening , night  "
                "2. Highlight local dishes and restaurants, with approximate costs. "
                "3. Mention accommodations with price ranges. "
                "4. Include transport options between locations (bus, train, taxi, etc.) with approximate fares. "
                "5.tips to be safe and avoid scams  "
                "6.at the very end add an estimation for the trip . "
                "7. Include any unique local tips or must-see spots. "
                
                "8. Be concise, clear, and engaging."
            )
        }

]

    # ==============================
    # Query LLM
    # ==============================
    start_time = time.time()
    answer = answer_question(messages)
    llm_time = time.time() - start_time
    print(f"⏱️ LLM response took {llm_time:.2f} seconds\n")

    # ==============================
    # Display answer
    # ==============================
    print("\n💡 Answer from LLM:\n")
    print(answer)

if __name__ == "__main__":
    main()
