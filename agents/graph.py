"""
agents/graph.py
Wires the Planner -> Executor -> Writer -> Critic agents into a LangGraph
StateGraph with a conditional revision loop (Writer <-> Critic).

    START -> planner -> executor -> writer -> critic --PASS--> END
                                        ^___________REVISE_______|
"""
from langgraph.graph import StateGraph, START, END
from agents.state import ResearchState
from agents.planner import planner_node
from agents.executor import executor_node
from agents.writer import writer_node
from agents.critic import critic_node, route_after_critic


def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("writer", writer_node)
    graph.add_node("critic", critic_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "writer")
    graph.add_edge("writer", "critic")
    graph.add_conditional_edges(
        "critic", route_after_critic, {"writer": "writer", "end": END}
    )

    return graph.compile()


def run_research(topic: str) -> dict:
    """Convenience entrypoint: runs the full graph for a topic and returns state."""
    app = build_graph()
    initial_state = {
        "topic": topic,
        "sub_questions": [],
        "evidence": [],
        "draft_report": "",
        "critique": "",
        "citations": [],
        "revision_count": 0,
        "final_report": "",
    }
    final_state = app.invoke(initial_state)
    return final_state


if __name__ == "__main__":
    import sys
    topic = " ".join(sys.argv[1:]) or "Impact of agentic AI systems on software engineering productivity"
    result = run_research(topic)
    print("\n=== FINAL REPORT ===\n")
    print(result["final_report"])
