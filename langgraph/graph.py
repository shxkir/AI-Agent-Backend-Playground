"""
Example LangGraph workflow for EdgeLink.

This script defines a simple graph with three nodes: classify, retrieve and answer.
It is not executed at runtime by default; instead it demonstrates how to set up
a LangGraph state graph.  You can expand upon this skeleton to build a full
agentic workflow with multiple agents such as ManagerAgent, RetrieverAgent and
CoderAgent as described in the design blueprint.
"""

from langgraph import StateGraph, END  # type: ignore[import]


def classify_intent(state):
    """Classify intent (stub)."""
    query = state.get("query", "")
    if query.lower().startswith("code"):
        state["intent"] = "task"
    else:
        state["intent"] = "ask"
    return state


def retrieve(state):
    """Stub retrieval node."""
    state["retrieved"] = ["doc1", "doc2"]
    return state


def generate_answer(state):
    """Stub answer generation."""
    query = state.get("query", "")
    state["answer"] = f"Answer to: {query}"
    return state


def build_graph():
    """
    Build and return a LangGraph state machine.

    The graph routes the request to different nodes based on the
    classified intent and produces an answer.  Replace the stubbed
    nodes with calls to your retrieval and LLM functions.
    """
    graph = StateGraph()
    graph.add_node("classify", classify_intent)
    graph.add_node("retrieve", retrieve)
    graph.add_node("answer", generate_answer)
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)
    return graph.compile()


if __name__ == "__main__":
    # Example invocation of the compiled graph
    app = build_graph()
    state = {"query": "What is the mission of EdgeUp?"}
    result = app.invoke(state)
    print(result)
