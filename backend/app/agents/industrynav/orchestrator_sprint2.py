# in backend/app/agents/industrynav/orchestrator_sprint2.py
from crewai import Crew, Process, Task

# --- 1. Import All Your Agents ---
from .agent_paper_parser import paper_parser_agent        # Aziz's Agent
from .agent_research_expander import research_expansion_agent  # <-- Your Agent
from .agent_novelty_assessor import novelty_assessor_agent   # Taher's Agent
from .agent_industry_mapper import industry_mapper_agent     # Linda's Agent
from .agent_report_generator import report_generator_agent   # Taher's Agent

# --- 2. Define All Your Tasks Here ---

# Task for Aziz's Agent
task_analyze_paper = Task(
    description=(
        "Analyze the provided research paper content. Extract the main topic, "
        "a list of key methodologies/keywords, and all mentioned references. "
        "Input Paper Content: {paper_content}"
    ),
    expected_output=(
        "A Python dictionary containing three keys: "
        "'main_topic': A short string describing the paper's main topic. "
        "'keywords': A list of key terms and technologies. "
        "'references': A list of strings, each being a cited paper title."
    ),
    agent=paper_parser_agent
)

# Task for Your (Idriss's) Agent
task_expand_research = Task(
    description=(
        "Using the analysis from the paper parser, find related literature. "
        "Specifically, find 3-5 highly-cited 'Survey Papers' on the main topic "
        "and 1-2 'Foundational Papers' that are heavily referenced."
    ),
    expected_output=(
        "A markdown section with two sub-headings: "
        "## Key Survey Papers\n- [Title](URL) - Brief justification.\n"
        "## Foundational Papers\n- [Title](URL) - Brief justification."
    ),
    agent=research_expansion_agent,
    context=[task_analyze_paper]  # <-- Depends on Aziz's task
)

# Task for Taher's Agent
task_score_novelty = Task(
    description=(
        "Based on the original paper's keywords and the related literature found, "
        "assess the paper's novelty. Is it truly new, or is it an incremental "
        "improvement? Compare it to the foundational papers."
    ),
    expected_output=(
        "A 'Novelty Score' (e.g., 8/10) and a 2-paragraph justification "
        "explaining *why* it is or isn't novel."
    ),
    agent=novelty_assessor_agent,
    context=[task_analyze_paper, task_expand_research] # Depends on you and Aziz
)

# Task for Linda's Agent
task_find_applications = Task(
    description=(
        "Given the paper's main topic and keywords (e.g., 'AI for climate modeling'), "
        "identify 3-5 potential real-world industry applications. "
        "Think about what companies or sectors could use this technology."
    ),
    expected_output=(
        "A markdown section: ## Potential Industry Applications\n"
        "1. **Industry (e.g., Renewable Energy):** How it can be applied.\n"
        "2. **Industry (e.g., Logistics):** How it can be applied."
    ),
    agent=industry_mapper_agent,
    context=[task_analyze_paper] # Depends on Aziz
)

# Task for Taher's Other Agent
task_generate_report = Task(
    description=(
        "Compile all the inputs from the previous agents (analysis, related papers, "
        "novelty score, and industry applications) into a single, "
        "comprehensive, and well-structured markdown report."
    ),
    expected_output=(
        "The final, complete markdown report, starting with # Industry Navigator Report"
    ),
    agent=report_generator_agent,
    context=[
        task_analyze_paper,
        task_expand_research,
        task_score_novelty,
        task_find_applications
    ] # This task depends on EVERYONE
)

# --- 3. Assemble the Crew (This is the "Orchestration"!) ---
industry_nav_crew = Crew(
    agents=[
        paper_parser_agent,
        research_expansion_agent,
        novelty_assessor_agent,
        industry_mapper_agent,
        report_generator_agent
    ],
    tasks=[
        task_analyze_paper,
        task_expand_research,
        task_score_novelty,
        task_find_applications,
        task_generate_report  # <-- The report generator runs last
    ],
    process=Process.sequential,  # Agents work one after another
    verbose=2
)

# --- 4. Create the Function for the API ---
# The API file (endpoints.py) will import THIS function.
def run_industry_nav_crew(paper_content: str):
    """
    Kicks off the Industry Navigator crew with the paper content.
    """
    inputs = {
        'paper_content': paper_content,
    }

    print("🚀 Kicking off the Industry Navigator Crew...")
    # Using .kickoff() is simple and good for a hackathon
    result = industry_nav_crew.kickoff(inputs=inputs)
    
    print("--- 🏁 CREW EXECUTION FINISHED ---")
    return result