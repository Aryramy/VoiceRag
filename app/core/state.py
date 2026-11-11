from typing import Annotated, Sequence, TypedDict, Any, List, Dict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgenticVoiceState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    raw_query: str
    rewritten_query: str
    retrieved_context: List[Dict[str, Any]]
    is_context_relevant: bool
    final_answer: str
    citations: List[Dict[str, Any]]
    detected_language: str
