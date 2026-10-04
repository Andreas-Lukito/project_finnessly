import json
from datetime import datetime

from crewai.tools import tool

from pathlib import Path

PROJECT_PATH = Path(__file__).resolve().parents[2]

REPORTS_PATH = PROJECT_PATH / "reports"

REPORT_TEMPLATE = """# Financial News Report

**Topic:** {topic}
**Generated:** {generated_at}

## News Summary

{summary}

## Sentiment Analysis

{sentiment}

## Final Overview

{overview}
"""


def render_markdown_report(
    topic: str,
    summary: str,
    sentiment: str,
    overview: str,
    filename: str | None = None,
) -> str:
    """
    Render a structured financial report as a markdown file.

    Args:
        topic (str): The topic of the report.
        summary (str): The news summary content.
        sentiment (str): The sentiment analysis results content.
        overview (str): The final overview content.
        filename (str | None): Optional filename. Defaults to a timestamped name.

    Returns:
        str: The path of the written markdown file.
    """
    REPORTS_PATH.mkdir(parents=True, exist_ok=True)

    filename = Path(filename).name if filename else f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

    if not filename.endswith(".md"):
        filename += ".md"

    report_path = REPORTS_PATH / filename

    report = REPORT_TEMPLATE.format(
        topic=topic,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        summary=summary,
        sentiment=sentiment,
        overview=overview,
    )

    report_path.write_text(report, encoding="utf-8")

    return str(report_path)


@tool("Markdown Report Renderer")
def markdown_render_tool(
    topic: str,
    summary: str,
    sentiment: str,
    overview: str,
    filename: str | None = None,
) -> str:
    """Render a structured financial news report as a markdown file and save it to the reports directory. Provide the topic, a news summary, the sentiment analysis results, and a final overview. Optionally provide a filename."""
    return render_markdown_report(topic, summary, sentiment, overview, filename)


if __name__ == "__main__":
    result = markdown_render_tool.run(**{
        "topic": "NVIDIA stock",
        "summary": "NVIDIA reported record revenue.",
        "sentiment": "Overall sentiment: positive.",
        "overview": "Positive outlook.",
    })
    print(result)