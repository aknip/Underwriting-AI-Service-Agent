import asyncio
from dataclasses import dataclass, field
import logging
import os
import sys
from typing import Any, Callable, List, Optional

from bs4 import BeautifulSoup
import openai
import httpx
import praw
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.tools import ToolDefinition


# --- General Configuration ---
USER_AGENT = "MyAgent/1.0"
ANALYZE_URL = "https://www.github.com"
RETRIES = 3
logging.basicConfig(level=logging.INFO)
sys.excepthook = lambda *args: None


# --- Reddit API Configuration ---
REDDIT_MAX_INSIGHTS = 5
REDDIT_MAX_INSIGHT_LENGTH = 400
REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET")


@dataclass
class SwotAgentDeps:
    """Dependencies for the Undweriting Agent."""

    request: Optional[Any] = None
    update_status_func: Optional[Callable] = None
    tool_history: List[str] = field(default_factory=list)
    try:
        reddit: praw.Reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=USER_AGENT,
            check_for_async=False,
        )
    except praw.exceptions.PRAWException as e:
        reddit = None
        logging.info(
            f"Reddit client not initialized. Please set the REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET environment variables: {e}"
        )
    try:
        client = openai.OpenAI()
    except ValueError as e:
        client = None
        logging.info(
            f"OpenAI client not initialized. Check API key: {e}"
        )    


class SwotAnalysis(BaseModel):
    """Represents a SWOT analysis with strengths, weaknesses, opportunities, threats, and an overall analysis."""

    strengths: List[str] = Field(
        description="Internal strengths of the product/service"
    )
    weaknesses: List[str] = Field(
        description="Internal weaknesses of the product/service"
    )
    opportunities: List[str] = Field(description="External opportunities in the market")
    threats: List[str] = Field(description="External threats in the market")
    analysis: str = Field(
        description="A comprehensive analysis explaining the SWOT findings and their implications"
    )
    headcount: str = Field(
        description="Number of employees"
    )


async def report_tool_usage(
    ctx: RunContext[SwotAgentDeps], tool_def: ToolDefinition
) -> ToolDefinition:
    """Reports tool usage and results to update_status_func."""

    if tool_def.name in ctx.deps.tool_history:
        # Tool has already been used and reported
        return tool_def

    if ctx.deps.update_status_func:
        await ctx.deps.update_status_func(
            ctx.deps.request,
            f"Using tool: {tool_def.name}...",
        )
        ctx.deps.tool_history.append(tool_def.name)

    return tool_def


# Pydantic AI agent
swot_agent = Agent(
    model = OpenAIModel(
        'gpt-4o'
    ),
    deps_type=SwotAgentDeps,
    result_type=SwotAnalysis,
    system_prompt="""
        You are a strategic business analyst tasked with performing SWOT analysis.
        Analyze the given URL, identify internal strengths and weaknesses,
        and evaluate external opportunities and threats based on market conditions
        and competitive landscape. Use community insights to validate findings.

        For each category:
        - Strengths: Focus on internal advantages and unique selling points
        - Weaknesses: Identify internal limitations and areas for improvement
        - Opportunities: Analyze external factors that could be advantageous
        - Threats: Evaluate external challenges and competitive pressures
        - Headcount: Report the number of employees

        Provide a detailed analysis that synthesizes these findings into actionable insights.

        Answer everything in German language. 
    """,
    retries=RETRIES,
)


# custom validator for the Pydantic Agent swot_agent 
@swot_agent.result_validator
def validate_analysis(
    _ctx: RunContext[SwotAgentDeps], value: SwotAnalysis
) -> SwotAnalysis:
    """Validates the SWOT analysis for completeness and quality."""
    issues = []

    # Check minimum number of points in each category
    min_points = 2
    categories = {
        "Strengths": value.strengths,
        "Weaknesses": value.weaknesses,
        "Opportunities": value.opportunities,
        "Threats": value.threats,
    }

    for category_name, points in categories.items():
        if len(points) < min_points:
            issues.append(
                f"{category_name}: Should have at least {min_points} points. Currently has {len(points)}."
            )

    # Check analysis length
    min_analysis_length = 100
    if len(value.analysis) < min_analysis_length:
        issues.append(
            f"Analysis should be at least {min_analysis_length} characters. Currently {len(value.analysis)} characters."
        )

    if issues:
        logging.info(f"Validation issues: {issues}")
        raise ModelRetry("\n".join(issues))

    return value


# --- Tools ---
# defined by decorators for the Pydantic Agent swot_agent 
@swot_agent.tool(prepare=report_tool_usage)
async def fetch_website_content(_ctx: RunContext[SwotAgentDeps], url: str) -> str:
    """Fetches the HTML content of the given URL."""
    logging.info(f"Fetching website content for: {url}")
    async with httpx.AsyncClient(follow_redirects=True) as http_client:
        try:
            response = await http_client.get(url)
            response.raise_for_status()
            html_content = response.text
            soup = BeautifulSoup(html_content, "html.parser")
            text_content = soup.get_text(separator=" ", strip=True)
            print('')
            print(text_content)
            print('')
            return text_content
        except httpx.HTTPError as e:
            logging.info(f"Request failed: {e}")
            raise


# pylint: disable=W0718
@swot_agent.tool(prepare=report_tool_usage)
async def analyze_competition(
    ctx: RunContext[SwotAgentDeps],
    product_name: str,
    product_description: str,
) -> str:
    """Analyzes the competition for the given company or product."""
    logging.info(f"Analyzing competition for: {product_name}")

    prompt = f"""
    You are a competitive analysis expert. Analyze the competition for the following company or product:
    Name: {product_name}
    Description: {product_description}

    Provide a detailed analysis of:
    1. Key competitors and their market position
    2. Competitive advantages and disadvantages
    3. Market trends and potential disruptions
    4. Entry barriers and competitive pressures

    Answer everything in German language. 
    """

    if not ctx.deps.client:
        logging.info("Error: AI client not initialized.")
        return ""
    try:
        response = await ctx.deps.client.chat.completions.create(
            messages=[
                {
                "role": "user",
                "content": prompt,
                }
            ],
            model="gpt-4o-mini",
            max_tokens=1024,
            #temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.info(f"Error analyzing competition: {e}")
        return f"Error analyzing competition: {e}"


@swot_agent.tool(prepare=report_tool_usage)
async def get_reddit_insights(
    ctx: RunContext[SwotAgentDeps],
    query: str,
    subreddit_name: str = "googlecloud",
) -> str:
    """Gathers insights from a specific subreddit related to a query using PRAW."""
    logging.info(f"Getting Reddit insights from r/{subreddit_name} for query: {query}")
    if not ctx.deps.reddit:
        logging.info("Error: Reddit client not initialized.")
        return ""
    try:
        subreddit = ctx.deps.reddit.subreddit(subreddit_name)
        search_results = subreddit.search(
            query, limit=REDDIT_MAX_INSIGHTS, sort="relevance", time_filter="all"
        )

        insights = []
        for post in search_results:
            insights.append(
                f"Title: {post.title}\n"
                f"URL: {post.url}\n"
                f"Content: {post.selftext[:REDDIT_MAX_INSIGHT_LENGTH]}...\n"
            )
        return "\n".join(insights)
    except praw.exceptions.PRAWException as e:
        logging.info(f"Error fetching Reddit data: {e}")
        return f"Error fetching Reddit data: {e}"


# pylint: disable=W0718
async def run_agent(
    url: str = ANALYZE_URL,
    deps: SwotAgentDeps = SwotAgentDeps(),
) -> SwotAnalysis | Exception:
    """
    Runs the SWOT analysis agent.

    Args:
        url: The URL to analyze.
        deps: The dependencies for the agent.

    Returns:
        The SWOT analysis result or an exception if an error occurred.
    """

    try:
        deps.tool_history = []
        result = await swot_agent.run(
            f"Perform a comprehensive SWOT analysis for this product: {url}",
            deps=deps,
        )
        logging.info(f"Agent result: {result}")

        # Send the final result to the UI via update_status_func
        if deps.update_status_func:
            await deps.update_status_func(deps.request, "Analysis Complete")

        return result.data
    except Exception as e:
        logging.exception(f"Error during agent run: {e}")

        # Send the error to the UI via update_status_func
        if deps.update_status_func:
            await deps.update_status_func(deps.request, f"Error: {e}")

        return e


if __name__ == "__main__":
    data = asyncio.run(run_agent())
    logging.info(data)
