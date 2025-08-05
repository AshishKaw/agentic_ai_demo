from textwrap import dedent
from phi.agent import Agent
import phi.api
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
from phi.model.groq import Groq

import os
import phi
from phi.playground import Playground, serve_playground_app
# Load environment variables from .env file
load_dotenv()

phi.api=os.getenv("PHI_API_KEY")

API_KEY = "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"

model_id = "meta-llama/Llama-4-Maverick-17B-128E-Instruct"

## web search agent
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for the information",
    description="You are a financial analyst. This agent can analyze financial data and provide insights.",
    model=Groq(id=model_id, api_key=API_KEY),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tools_calls=True,
    markdown=True,
)

## Financial agent
finance_agent = Agent(
    name="Finance AI Agent",
    model=Groq(id=model_id, api_key=API_KEY),
    role="Analyze financial data",
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            company_news=True,
        ),
    ],
    instructions=dedent("""\
        You are a seasoned Wall Street analyst with deep expertise in market analysis! 📊

        Follow these steps for comprehensive financial analysis:
        1. Market Overview
           - Latest stock price
           - 52-week high and low
        2. Financial Deep Dive
           - Key metrics (P/E, Market Cap, EPS)
        3. Professional Insights
           - Analyst recommendations breakdown
           - Recent rating changes

        4. Market Context
           - Industry trends and positioning
           - Competitive analysis
           - Market sentiment indicators

        Your reporting style:
        - Begin with an executive summary
        - Use tables for data presentation
        - Include clear section headers
        - Add emoji indicators for trends (📈 📉)
        - Highlight key insights with bullet points
        - Compare metrics to industry averages
        - Include technical term explanations
        - End with a forward-looking analysis

        Risk Disclosure:
        - Always highlight potential risk factors
        - Note market uncertainties
        - Mention relevant regulatory concerns

        Use tables to display the data
    """),
    show_tool_calls=True,
    markdown=True,
)

multi_ai_agent = Agent(
    name="Stock Analysis Agent",
    team=[web_search_agent, finance_agent],
    model=Groq(id=model_id, api_key=API_KEY),
    instructions=[
        "Use the web search agent to find information about the company and the financial agent to find information about the stock.",
        "Use tables to display the data",
    ],
    show_tool_calls=True,
    markdown=True,
)

app=Playground(agents=[multi_ai_agent]).get_app()
if __name__ == "__main__":
    serve_playground_app("mybot:app",host="0.0.0.0",reload=True)


