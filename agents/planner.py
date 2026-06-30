"""
agents/planner.py
Planner node: decomposes the user's topic into 3-5 targeted sub-questions
that the Executor agent will research independently.
"""
import json
import re
from agents.llm import get_llm
from agents.state import ResearchState

PLANNER_PROMPT = """You are a research planning agent. Given a research topic, \
break it down into 3-5 specific, non-overlapping sub-questions that together \
would produce a thorough report on the topic.

Topic: {topic}

Respond ONLY with a JSON array of strings, e.g. ["question 1", "question 2", ...]. \
No preamble, no markdown fences."""


def planner_node(state: ResearchState) -> dict:
    llm = get_llm(temperature=0.2)
    prompt = PLANNER_PROMPT.format(topic=state["topic"])
    response = llm.invoke(prompt)
    text = response.content.strip()

    # defensive parsing in case the model wraps in fences anyway
    text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        sub_questions = json.loads(text)
        assert isinstance(sub_questions, list)
    except Exception:
        # fallback: split by lines if JSON parsing fails
        sub_questions = [l.strip("- ").strip() for l in text.split("\n") if l.strip()]

    return {"sub_questions": sub_questions[:5], "revision_count": 0}
