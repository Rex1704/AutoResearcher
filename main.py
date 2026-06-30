"""
main.py
FastAPI backend exposing the multi-agent research system as a REST API.

Run with: uvicorn main:app --reload
"""
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agents.graph import run_research

app = FastAPI(
    title="AutoResearcher",
    description="Multi-agent (Planner-Executor-Writer-Critic) autonomous research system built with LangGraph",
    version="0.1.0",
)


class ResearchRequest(BaseModel):
    topic: str


class ResearchResponse(BaseModel):
    topic: str
    sub_questions: list[str]
    report: str
    citations: list[dict]
    revision_cycles: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/research", response_model=ResearchResponse)
def research(req: ResearchRequest):
    if not req.topic or not req.topic.strip():
        raise HTTPException(status_code=400, detail="topic must not be empty")

    try:
        result = run_research(req.topic.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"research pipeline failed: {e}")

    return ResearchResponse(
        topic=req.topic,
        sub_questions=result.get("sub_questions", []),
        report=result.get("final_report") or result.get("draft_report", ""),
        citations=result.get("citations", []),
        revision_cycles=result.get("revision_count", 0),
    )
