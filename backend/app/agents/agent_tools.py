# in backend/app/agents/agent_tools.py

# Import the necessary tool decorator from crewai
from crewai import tool
# Import the community library for DuckDuckGo
from duckduckgo_search import DDGS

# Define the free search tool using the @tool decorator
# The function name 'search_tool' is what your agent will import.
@tool("Academic Search Tool (DuckDuckGo)")
def search_tool(query: str) -> str:
    """
    Searches the web for relevant academic papers, surveys, or foundational
    research using the free DuckDuckGo search engine.
    
    The tool takes a search query and returns the top 5 relevant snippets 
    and URLs, formatted cleanly for the LLM.
    """
    try:
        # Use DDGS().text for general text search results
        # We limit the results to 5 to avoid overwhelming the LLM context window
        results = DDGS().text(query, max_results=5)
        
        if not results:
            return "No relevant search results found for the query. The agent must proceed based only on internal knowledge."

        # Format the results into a clean, structured string for the LLM
        formatted_results = []
        for i, r in enumerate(results):
            formatted_results.append(
                f"--- Result {i+1} ---\n"
                f"Title: {r.get('title', 'N/A')}\n"
                f"Snippet: {r.get('snippet', 'N/A')}\n"
                f"URL: {r.get('href', 'N/A')}\n"
            )
        
        return "\n".join(formatted_results)

    except Exception as e:
        # Critical error handling for the agent
        return f"An error occurred during the DuckDuckGo search: {e}. The search tool failed."