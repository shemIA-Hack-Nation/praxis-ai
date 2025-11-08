# in backend/app/agents/papergen/orchestrator_sprint1.py
from crewai import Crew, Process, Task

# --- 1. Import ALL agents for Sprint 1 ---
from .agent_notebook_parser import notebook_parser_agent
from .agent_methodology_writer import methodology_writer_agent
from .agent_results_writer import results_writer_agent
from .agent_literary import literary_agent
from .agent_formatter import formatter_agent
from .agent_illustration_critic import illustration_critic_agent  # <-- ADD YOUR AGENT

# --- 2. Define ALL tasks for Sprint 1 ---
# (Tasks for parser, methodology, results... already defined by others)
# ...


task_critique_visuals = Task(
    description=(
        "Review the compiled draft of the research paper. Identify 3-5 "
        "key areas (like complex methodologies or data-heavy results) "
        "that require an illustration, graph, or diagram to improve clarity."
    ),
    expected_output=(
        "A list of specific suggestions for illustrations, including the "
        "section to place it and a brief description of what the visual should convey."
    ),
    agent=illustration_critic_agent,
    # This task should run AFTER the main sections are written,
    # but BEFORE the final formatting.
    context=[
        # task_write_methodology,  <-- Example dependencies
        # task_write_results
    ]
)

# ...

# --- 3. Assemble the Crew ---
# Make sure to add your agent and task to the lists
papergen_crew = Crew(
    agents=[
        notebook_parser_agent,
        methodology_writer_agent,
        results_writer_agent,
        illustration_critic_agent,  # <-- ADDED
        literary_agent,
        formatter_agent
    ],
    tasks=[
        # task_parse_notebook,
        # task_write_methodology,
        # task_write_results,
        task_critique_visuals,  # <-- ADDED
        # task_refine_literature,
        # task_format_paper
    ],
    process=Process.sequential,
    verbose=2
)

# --- 4. Create the API function ---
def run_papergen_crew(notebook_content: str, template: str):
    # ... (rest of the kickoff logic)
    pass