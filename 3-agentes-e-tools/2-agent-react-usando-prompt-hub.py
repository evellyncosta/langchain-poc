from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

#llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

@tool("calculator", return_direct=True)
def calculator(expression:str) -> str:
    """Evaluate a simple mathematical expression and returns the result."""
    try:
        result = eval(expression)
    except Exception as e:
        return f"Error: {e}"
    return str(result)

@tool("web_search_mock")
def web_search_mock(query: str) -> str:
    """Return the capital of a given country if it exists in the mock data."""
    data = {
        "Brazil": "Brasília",
        "France": "Paris",
        "Germany": "Berlin",
        "Italy": "Rome",
        "Spain": "Madrid",
        "United States": "Washington, D.C."
        
    }
    for country, capital in data.items():
        if country.lower() in query.lower():
            return f"The capital of {country} is {capital}."
    return "I don't know the capital of that country."

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)
tools = [calculator, web_search_mock]

prompt = hub.pull("hwchase17/react")

agent_chain = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent_chain,
    tools=tools,
    verbose=True,
    handle_parsing_errors="INvalid format. Either provide an Action with Action Input, or a Final Answer only",
)

#print(agent_executor.invoke({"input": "What is the capital of Iran?"}))
print(agent_executor.invoke({"input": "How much is 10 + 10? "}))