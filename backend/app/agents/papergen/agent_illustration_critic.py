# in backend/app/agents/papergen/agent_illustration_critic.py
from crewai import Agent,LLM

llm = LLM(model="gemini/gemini-2.5-flash", temperature=0.2)

illustration_critic_agent = Agent(
    role='Visual Enhancement Critic',
    goal=(
        "Critically review a draft of a research paper and identify the top 3-5 "
        "sections that are too complex, dense, or data-heavy and would "
        "significantly benefit from an illustration, diagram, or graph."
    ),
    backstory=(
        "You are an expert technical editor and visual designer. You know that "
        "a well-placed diagram can make a complex idea simple. Your job is not "
        "to create the image, but to tell the writers *where* an image is "
        "desperately needed and *what* it should show."
    ),
    tools=[],  # This agent just uses its LLM reasoning
    allow_delegation=False,
    verbose=True,
    llm=llm
)