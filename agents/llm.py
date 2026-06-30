"""
agents/llm.py
Thin wrapper around langchain-groq so every agent node shares one config.
"""
import os
from langchain_groq import ChatGroq

_LLM = None


def get_llm(temperature: float = 0.3):
    global _LLM
    model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    if _LLM is None:
        _LLM = ChatGroq(model=model, temperature=temperature)
    return _LLM
