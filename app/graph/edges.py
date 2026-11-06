from app.core.state import AgenticVoiceState

def route_after_retrieval(state: AgenticVoiceState) -> str:
    """Conditional router: Directs flow to answer generation or strict fallback."""
    if not state.get("is_context_relevant", False):
        return "generate_answer"
    return "generate_answer"

