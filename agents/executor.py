"""
agents/executor.py
Executor node: for each sub-question, performs web search + scraping to
gather grounding evidence. This is the "tool-calling" agent in the pipeline.
"""
from agents.state import ResearchState
from tools.search_tools import gather_evidence


def executor_node(state: ResearchState) -> dict:
    all_evidence = []
    for question in state["sub_questions"]:
        evidence = gather_evidence(question, max_results=3)
        for e in evidence:
            e["sub_question"] = question
        all_evidence.extend(evidence)
    return {"evidence": all_evidence}
