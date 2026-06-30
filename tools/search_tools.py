"""
tools/search_tools.py
Web search + page scraping tools used by the Executor agent.
"""
import os
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from tenacity import retry, stop_after_attempt, wait_exponential

MAX_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", 5))
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; AutoResearcherBot/1.0)"}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def web_search(query: str, max_results: int = MAX_RESULTS) -> list[dict]:
    """Search the web via DuckDuckGo and return a list of {title, url, snippet}."""
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            })
    return results


def scrape_page(url: str, max_chars: int = 6000) -> str:
    """Fetch a URL and return cleaned visible text, truncated to max_chars."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=8)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "form"]):
            tag.decompose()

        text = " ".join(soup.get_text(separator=" ").split())
        return text[:max_chars]
    except Exception as e:
        return f"[scrape_failed: {e}]"


def gather_evidence(query: str, max_results: int = MAX_RESULTS) -> list[dict]:
    """Search + scrape top results. Returns list of {title, url, content}."""
    hits = web_search(query, max_results=max_results)
    evidence = []
    for h in hits:
        content = scrape_page(h["url"])
        if content and not content.startswith("[scrape_failed"):
            evidence.append({"title": h["title"], "url": h["url"], "content": content})
        else:
            # fall back to the snippet if scraping fails
            evidence.append({"title": h["title"], "url": h["url"], "content": h["snippet"]})
    return evidence
