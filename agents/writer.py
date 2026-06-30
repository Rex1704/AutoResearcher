"""
agents/writer.py
Writer node: builds a FAISS index over collected evidence (RAG), retrieves
the most relevant chunks per sub-question, and drafts a cited report.
"""
from agents.llm import get_llm
from agents.state import ResearchState
from tools.rag_store import EvidenceStore

WRITER_PROMPT = """You are a research report writer. Using ONLY the evidence \
snippets provided below (each tagged with a source number), write a clear, \
well-structured report answering the topic.

Topic: {topic}

Evidence:
{evidence_block}

Instructions:
- Organize the report with headings per sub-question.
- Cite sources inline using [n] matching the evidence numbering.
- Do not invent facts not present in the evidence.
- End with a "Sources" section listing each [n] -> URL.
{revision_note}
Write the full report now."""


def writer_node(state: ResearchState) -> dict:
    store = EvidenceStore()
    store.add_evidence(state["evidence"])

    evidence_lines = []
    citations = []
    seen_urls = {}
    counter = 1

    for question in state["sub_questions"]:
        retrieved = store.query(question, k=4)
        for r in retrieved:
            if r["url"] not in seen_urls:
                seen_urls[r["url"]] = counter
                citations.append({"id": counter, "title": r["title"], "url": r["url"]})
                counter += 1
            sid = seen_urls[r["url"]]
            evidence_lines.append(f"[{sid}] (re: {question}) {r['text'][:500]}")

    evidence_block = "\n\n".join(evidence_lines) if evidence_lines else "No evidence retrieved."

    revision_note = ""
    if state.get("critique") and state["critique"] != "PASS":
        revision_note = f"\nIMPORTANT - address this editorial feedback from the previous draft:\n{state['critique']}\n"

    llm = get_llm(temperature=0.3)
    prompt = WRITER_PROMPT.format(
        topic=state["topic"], evidence_block=evidence_block, revision_note=revision_note
    )
    response = llm.invoke(prompt)

    return {"draft_report": response.content, "citations": citations}
