from langgraph.graph import StateGraph, END
from app.core.state import AgenticVoiceState
from app.graph.nodes import rewrite_query_node, retrieve_node, generate_answer_node
from app.graph.edges import route_after_retrieval

builder = StateGraph(AgenticVoiceState)
builder.add_node("rewrite_query", rewrite_query_node)
builder.add_node("retrieve", retrieve_node)
builder.add_node("generate_answer", generate_answer_node)

builder.set_entry_point("rewrite_query")
builder.add_edge("rewrite_query", "retrieve")
builder.add_conditional_edges("retrieve", route_after_retrieval)
builder.add_edge("generate_answer", END)

graph = builder.compile()

