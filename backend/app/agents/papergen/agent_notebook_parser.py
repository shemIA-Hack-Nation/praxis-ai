from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from pydantic import SecretStr
from backend.app.agents.agent_tools import parse_notebook
from backend.app.core import config

# Note: The create_agent function uses built-in ReAct prompting, so we don't need a custom prompt

def get_notebook_parser_agent():
    tools = [parse_notebook]
    # Convert API key to SecretStr if it exists
    api_key = SecretStr(config.GROQ_API_KEY) if config.GROQ_API_KEY else None
    llm = ChatGroq(api_key=api_key, temperature=0, model="llama-3.1-8b-instant")
    
    agent = create_agent(llm, tools)
    
    return agent

def run_notebook_parser(notebook_path: str):
    agent = get_notebook_parser_agent()
    message = HumanMessage(content=f"Parse the notebook at this path: {notebook_path}")
    result = agent.invoke({"messages": [message]})
    
    # Extract the final message from the agent response
    if "messages" in result and result["messages"]:
        return result["messages"][-1].content
    else:
        return "No output from agent"

if __name__ == '__main__':
    import sys
    import os
    import json

    if len(sys.argv) > 1:
        notebook_path = sys.argv[1]
        if os.path.exists(notebook_path):
            print(f"Parsing notebook: {notebook_path}")
            try:
                parsed_content = run_notebook_parser(os.path.abspath(notebook_path))
                print("\nParsed Content:")
                # The output from the tool is a string representation of a list, so we evaluate it.
                # In a real application, you might want to ensure the output is directly JSON.
                print(json.dumps(eval(parsed_content), indent=2))
            except Exception as e:
                print(f"An error occurred: {e}")
        else:
            print(f"Error: Notebook file not found at '{notebook_path}'")
    else:
        print("Usage: python -m backend.app.agents.papergen.agent_notebook_parser <path_to_notebook>.ipynb")

