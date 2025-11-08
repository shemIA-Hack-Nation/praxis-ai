# in backend/app/agents/industrynav/agent_research_expander.py
from crewai import Agent,LLM

# Import the tool from the central tools file
from app.agents.agent_tools import search_tool

# Define the LLM we'll be working with
llm = LLM(model="gemini/gemini-2.5-flash", temperature=0.2)

# --- IDRISS'S SPRINT 2 AGENT ---
research_expansion_agent = Agent(
    role='Research Expansion Specialist',
    goal=(
        "Find related surveys, foundational papers, and citing papers based on "
        "an analysis of a given research paper."
    ),
    backstory=(
        "A master librarian and academic researcher who excels at connecting "
        "one paper to the entire web of scientific literature. You know exactly "
        "how to use search terms to find the most relevant and impactful related work."
    ),
    tools=[search_tool],  # This agent can use the internet!
    allow_delegation=False,
    verbose=True,
    llm=llm
)