import asyncio
from app.graph.workflow import graph
from app.config import settings
from app.worker.audio_pipeline import strip_citations_for_tts

async def run_tests():
    print("=== Testing Agentic RAG Voice Graph ===")

    test_queries = [
        "What is your return and refund policy?",
        "Can I get a replacement if I drop my hardware item by accident?",
        "What is the capital of Mars?"  # Should trigger strict fallback
    ]

    for q in test_queries:
        print(f"\n--- Query: '{q}' ---")
        state = {
            "raw_query": q,
            "messages": []
        }
        config = {
            "configurable": {
                "thread_id": "test-session-1",
                "openrouter_key": settings.OPENROUTER_API_KEY
            }
        }
        result = await graph.ainvoke(state, config=config)
        raw_answer = result.get("final_answer", "")
        spoken_text = strip_citations_for_tts(raw_answer)

        print(f"Rewritten Query: {result.get('rewritten_query')}")
        print(f"Context Relevant: {result.get('is_context_relevant')}")
        print(f"Raw Output (with citations): {raw_answer}")
        print(f"Spoken Output (clean TTS): {spoken_text}")

if __name__ == "__main__":
    asyncio.run(run_tests())

