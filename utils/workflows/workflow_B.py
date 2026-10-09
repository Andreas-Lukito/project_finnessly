import torch
import wandb
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from crewai import Agent, Task, Crew, Process
from tqdm import tqdm
import os

# Project Config
from pathlib import Path
import sys

## Project root
PROJECT_PATH = Path("..").resolve()

## Add project root to Python path
sys.path.append(str(PROJECT_PATH))

## Internal tools
from utils.tools.search_tool import search_searxng
from utils.tools.sentiment_tool import analyze_sentiment
from utils.tools.markdown_render_tool import (
    markdown_render_tool,
    render_markdown_report,
    REPORTS_PATH,
)

reports_path = str(REPORTS_PATH)

## Environment variables
from dotenv import load_dotenv

load_dotenv(PROJECT_PATH / ".env")

## LLM
from crewai import LLM

# LLM Config
llm = LLM(
    model="openai/gpt-5-mini"
)

# Tools
from crewai.tools import tool

search_tool = search_searxng

@tool("FinBERT Sentiment Analysis")
def sentiment_tool(text: str, model_type: str = "finbert") -> str:
    """Analyze the sentiment of a news article text using FinBERT or BERT. Returns label and confidence score."""
    # Debug
    print(f"Analyzing sentiment for text: {text[-100:]}... with model: {model_type}")
    return str(analyze_sentiment(text, model_type))

markdown_render_tool = markdown_render_tool

#Agents
news_gathering = Agent(
    role="news_gathering",
    goal="Provide news latest articles about stock market and finance prompted by the user. Use the search tool to gather relevant information provide 5 until 10 news articles.",
    backstory="You are a news gathering agent that collects the latest news articles about stock market and finance based on user prompts. You will provide relevant and up-to-date information to assist users in making informed decisions.",
    tools=[search_tool],
    llm=llm,
)

news_sentiment = Agent(
    role="news_sentiment",
    goal="Use the sentiment tool to analyze the sentiment of news articles about stock market and finance prompted by the user. Provide a summary of the sentiment analysis results. the tool would output a sentiment of neutral, positive, or negative.",
    backstory="You are a sentiment analysis agent that analyzes the sentiment of news articles about stock market and finance based on user prompts. You will provide a summary of the sentiment analysis results to assist users in understanding the overall sentiment of the news articles.",
    tools=[sentiment_tool],
    llm=llm,
)

news_summarizing = Agent(
    role="news_summarizing",
    goal="summarize the news articles with the sentiment analysis results and provide a summary of the news articles about stock market and finance prompted by the user.",
    backstory="You are a news gathering agent that collects the latest news articles about stock market and finance based on user prompts. You will provide relevant and up-to-date information to assist users in making informed decisions.",
    llm=llm,
)

news_reporting = Agent(
    role="news_reporting",
    goal="Compile the summarized news articles with sentiment analysis results and provide a final report of the news articles about stock market and finance prompted by the user.",
    backstory="You are a news gathering agent that collects the latest news articles about stock market and finance based on user prompts. You will provide relevant and up-to-date information to assist users in making informed decisions.",
    tools=[markdown_render_tool],
    llm=llm,
)

task_gathering = Task(
    description=(
        "Search for the latest news articles about '{topic}' using the search tool. "
        "Collect relevant article titles, URLs, and content snippets worth analyzing."
    ),
    expected_output=(
        "A structured list of the latest relevant news articles about '{topic}', "
        "including title, URL, and content for each article."
    ),
    agent=news_gathering,
)

task_sentiment = Task(
    description=(
        "For each news article gathered about '{topic}', call the analyze_sentiment function "
        "(imported from utils.tools.sentiment_tool) on its text using model_type='finbert'. "
        "Aggregate the results into an overall sentiment summary."
    ),
    expected_output=(
        "A per-article sentiment breakdown (positive/negative/neutral with confidence) "
        "and an overall aggregated sentiment summary for '{topic}'."
    ),
    agent=news_sentiment,
    context=[task_gathering],
)

task_summarizing = Task(
    description=(
        "Summarize the gathered news articles about '{topic}' together with their sentiment analysis results "
        "into a clear, concise summary."
    ),
    expected_output=(
        "A well-structured summary of the latest news about '{topic}' that integrates "
        "the sentiment analysis findings."
    ),
    agent=news_summarizing,
    context=[task_gathering, task_sentiment],
)

task_reporting = Task(
    description=(
        "Compile the summarized news and sentiment results about '{topic}' into a final structured financial report, "
        "then use the markdown_render_tool (topic, summary, sentiment, overview) to save it as a markdown file. "
        "The tool always saves to the project's reports directory — do not pass a path in the filename."
    ),
    expected_output=(
        "A complete structured financial report about '{topic}' in markdown format, saved to disk."
    ),
    agent=news_reporting,
    context=[task_summarizing],
    output_file=os.path.join(reports_path, "financial_report_bert.md"),
)

crew = Crew(
    agents=[news_gathering, news_sentiment, news_summarizing, news_reporting],
    tasks=[task_gathering, task_sentiment, task_summarizing, task_reporting],
    process=Process.sequential,
    verbose=True,
)

async def workflow_B_run(topic: str) -> str:
    result = await crew.kickoff_async(inputs={"topic": topic})

    return str(result)