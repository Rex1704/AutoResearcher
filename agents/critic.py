"""
agents/critic.py
Critic node: evaluates the draft report for completeness, grounding, and
clarity. Returns PASS or specific revision feedback, driving the conditional
edge back to the writer (bounded by MAX_REVISION_CYCLES).
"""
import os
from agents.llm import get_llm
from agents.state import ResearchState

CRITIC_PROMPT = """You are a strict editorial critic reviewing a research report.

Topic: {topic}
Sub-questions it should address: {sub_questions}

Report:
{draft}

Check whether the report:
1. Addresses every sub-question
2. Is grounded in cited evidence (has [n] citations, not vague claims)
3. Is well organized and free of obvious contradictions

If the report is acceptable, respond with exactly: PASS
Otherwise, respond with: REVISE: <specific, actionable feedback in 2-3 sentences>"""

MAX_REVISIONS = int(os.getenv("MAX_REVISION_CYCLES", 2))


def critic_node(state: ResearchState) -> dict:
    llm = get_llm(temperature=0.1)
    prompt = CRITIC_PROMPT.format(
        topic=state["topic"],
        sub_questions=", ".join(state["sub_questions"]),
        draft=state["draft_report"],
    )
    response = llm.invoke(prompt).content.strip()

    revision_count = state.get("revision_count", 0)

    if response.upper().startswith("PASS") or revision_count >= MAX_REVISIONS:
        return {"critique": "PASS", "final_report": state["draft_report"]}

    return {"critique": response, "revision_count": revision_count + 1}


def route_after_critic(state: ResearchState) -> str:
    """Conditional edge: loop back to writer or finish."""
    return "end" if state["critique"] == "PASS" else "writer"
