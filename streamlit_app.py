from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from agents.graph import run_research

st.set_page_config(page_title="AutoResearcher", page_icon="", layout="wide")
st.title("AutoResearcher — Multi-Agent Research Assistant")
st.caption("Planner → Executor (web search + scrape) → Writer (RAG-grounded) → Critic loop, built with LangGraph")

topic = st.text_input("Research topic", placeholder="e.g. State of small language models in 2026")

if st.button("Run research", type="primary") and topic.strip():
    with st.spinner("Agents are researching... this can take 30-90s"):
        result = run_research(topic.strip())

    st.subheader("Sub-questions investigated")
    for q in result.get("sub_questions", []):
        st.markdown(f"- {q}")

    st.subheader("Report")
    st.markdown(result.get("final_report") or result.get("draft_report", ""))

    with st.expander("Revision cycles & sources"):
        st.write(f"Revision cycles used: {result.get('revision_count', 0)}")
        for c in result.get("citations", []):
            st.markdown(f"[{c['id']}] [{c['title']}]({c['url']})")
