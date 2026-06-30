"""
agents/state.py
Shared state object passed between nodes in the LangGraph.
"""
from typing import TypedDict, Annotated
import operator


class ResearchState(TypedDict):
    topic: str                                   # user's research topic
    sub_questions: list[str]                      # planner output
    evidence: Annotated[list[dict], operator.add]  # accumulated scraped evidence
    draft_report: str                              # writer output
    critique: str                                  # critic feedback
    citations: list[dict]                          # url/title pairs used
    revision_count: int
    final_report: str
