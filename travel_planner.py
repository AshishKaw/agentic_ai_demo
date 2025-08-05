from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.exa import ExaTools


API_KEY = "ADD KEY"

model_id = "meta-llama/Llama-4-Maverick-17B-128E-Instruct"

## web search agent
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for the information",
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
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            company_news=True,
        ),
    ],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

# multi agent application
multi_ai_agent = Agent(
    team=[web_search_agent, finance_agent],
    model=Groq(id=model_id, api_key=API_KEY),
    instructions=[
        "Use the web search agent to find information about the company and the financial agent to find information about the stock.",
        "Use tables to display the data",
    ],
    show_tool_calls=True,
    markdown=True,
)

multi_ai_agent.print_response("Summarize analyst recommendations and latest news for NVDA.", stream=True)
