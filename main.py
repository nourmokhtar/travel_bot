from query_handler import extract_location
from retrival import query_qdrant
from llm import answer_question
from search_agent import search_agent_fallback, save_fallback_to_qdrant
  # your fallback search agent
import asyncio
import time

# ==============================
# Main flow
# ==============================
async def main():
    while True:
        user_query = input("📝 Enter your travel query (or type 'exit' to quit): ")
        if user_query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break
        # Empty input check
        if not user_query:
            print("⚠️ Empty query. Please enter a question.")
            continue  # Skip this loop and prompt again
        
        location = extract_location(user_query)
        if location:
            print(f"🔍 Extracted location: {location}")
        else:
            print("⚠️ No location detected, querying globally...")

        # Retrieve context from Qdrant
        context_results = query_qdrant(user_query, location=location, top_k=5)
        # If context found, prepare LLM messages
        context_texts = "\n\n".join([r["text"] for r in context_results])
        
        # If no context found, fallback to online search
        if not context_results:
            print("🌐 No local data found, using fallback search agent...")
            import asyncio
            answer = await search_agent_fallback(user_query)  # <-- changed
            context_texts = answer  # No joining, since it's already text


            # Optionally, extract location info from user_query for KB
            extracted_location = extract_location(user_query)
            country, city = (None, None)
            if extracted_location and "-" in extracted_location:
                country, city = extracted_location.split("-", 1)
            elif extracted_location:
                country = extracted_location

            try:
                success = save_fallback_to_qdrant(
                    question=user_query,
                    answer=context_texts,
                    location_key=extracted_location,
                    country=country,
                    city=city
                )
                if success:
                    print("✅ Fallback result added successfully.")
                else:
                    print("⚠️ Fallback registration returned False.")
            except Exception as e:
                print(f"❌ Error registering fallback result: {e}")


            print("\n💡 Answer from fallback agent:\n")
            print(answer)
              # stop here since fallback handled the answer

        

        # ------------------------------
        # 4. Prepare messages for LLM
        # ------------------------------
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a world-class travel assistant. "
                    "You provide detailed, friendly, and practical travel advice. "
                    "Use ONLY the provided context to create personalized travel itineraries, "
                    "recommend local activities, accommodations, restaurants, dishes, transport options, "
                    "and provide estimated price ranges. "
                    "Tips to be safe and avoid scams. "
                    "Highlight must-see attractions. "
                    "At the very end add an estimation for the trip. "
                    "If context is missing, say you don't know."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Context (from travel database or fallback):\n{context_texts}\n\n"
                    f"User Question:\n{user_query}\n\n"
                    "Instructions: "
                    "1. Suggest a complete itinerary with day-by-day plan (morning, evening, night). "
                    "2. Highlight local dishes and restaurants with approximate costs. "
                    "3. Mention accommodations with price ranges. "
                    "4. Include transport options between locations (bus, train, taxi, etc.) with approximate fares. "
                    "5. Tips to be safe and avoid scams. "
                    "6. At the very end add an estimation for the trip. "
                    "7. Include any unique local tips or must-see spots. "
                    "8. Be concise, clear, and engaging."
                )
            }
        ]

        # ------------------------------
        # 5. Ask LLM
        # ------------------------------
        start_time = time.time()
        answer = answer_question(messages)
        llm_time = time.time() - start_time
        print(f"⏱️ LLM response took {llm_time:.2f} seconds\n")

        # ------------------------------
        # 6. Display answer
        # ------------------------------
        print("\n💡 Answer from LLM:\n")
        print(answer)

if __name__ == "__main__":
    asyncio.run(main())