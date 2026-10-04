import requests
import dotenv
from crewai.tools import tool
from scrapling.fetchers import Fetcher

SEARXNG_API_URL = dotenv.get_key(
    dotenv.find_dotenv(),
    "SEARXNG_URL"
) or "http://localhost:8080/search"

@tool("SearXNG Search")
def search_searxng(query: str) -> str:
    """
    Search the web using SearXNG and return relevant search results.
    """
    response = requests.get(
        SEARXNG_API_URL,
        params={
            "q": query,
            "format": "json",
            "language": "en"
        },
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    results = []

    for result in data.get("results", []):
        results.append(
            f"Title: {result.get('title', '')}\n"
            f"URL: {result.get('url', '')}\n"
            f"Content: {result.get('content', '')}"
        )

    return "\n\n".join(results)


@tool("News Article Scraper")
def search_full_articles(query: str, max_articles: int = 5) -> str:
    """
    Search the web using SearXNG and fetch the full article content
    for the top results. For each article, returns the title, URL, and
    the extracted full text (snippets only when full content fails).
    """
    response = requests.get(
        SEARXNG_API_URL,
        params={
            "q": query,
            "format": "json",
            "language": "en"
        },
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    results = []

    for result in data.get("results", [])[:max_articles]:
        url = result.get("url", "")
        title = result.get("title", "")
        snippet = result.get("content", "")

        scraped = scrape_article(url)

        if scraped["status"] == "success":
            content = scraped["content"]
        else:
            content = f"(full text unavailable: {scraped['reason']}; snippet: {snippet})"

        results.append(
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Content: {content}"
        )

    return "\n\n".join(results)

def scrape_article(url: str) -> dict:
    """Fetch and extract the main content from a news article."""

    try:
        page = Fetcher.get(url)

        if page.status != 200:
            return {
                "status": "failed",
                "url": url,
                "content": None,
                "reason": f"HTTP {page.status}"
            }

        content = page.markdown(main_content_only=True)

        if not content:
            return {
                "status": "failed",
                "url": url,
                "content": None,
                "reason": "No article content extracted"
            }
        lowered = content.lower()
        bot_markers = (
            "verify you are human",
            "enable javascript and cookies",
            "are you a robot",
            "bot detection",
            "captcha",
            "click allow to verify",
            "ddos protection by",
            "access denied",
            "just a moment",
        )

        if any(marker in lowered for marker in bot_markers):
            return {
                "status": "failed",
                "url": url,
                "content": None,
                "reason": "Content appears to be blocked by bot detection"
            }
        
        return {
            "status": "success",
            "url": url,
            "content": content
        }

    except Exception as e:
        return {
            "status": "failed",
            "url": url,
            "content": None,
            "reason": str(e)
        }

if __name__ == "__main__":
    result = search_searxng.run("NVIDIA stock news")
    print(result)