from crewai import Crew, Process, Task
from dotenv import load_dotenv

# Charger les variables d'environnement (pour le LLM)
load_dotenv()

# --- 1. Import ALL agents for Sprint 1 ---
# Note: Ces imports supposent que vos coéquipiers ont déjà défini ces agents.
from .agent_notebook_parser import notebook_parser_agent
from .agent_methodology_writer import methodology_writer_agent
from .agent_results_writer import results_writer_agent
from .agent_literary import literary_agent
from .agent_formatter import formatter_agent
from .agent_illustration_critic import illustration_critic_agent 

# --- 2. Define ALL tasks for Sprint 1 ---

# 2.1. Tâche Initiale (Coéquipier: Parser)
task_parse_notebook = Task(
    description=(
        "Analyze the raw Jupyter Notebook content and extract all data, code snippets, "
        "and preliminary analysis results necessary to write the paper sections. "
        "Focus on identifying the core methodology and the key findings. "
    ),
    expected_output="A structured JSON object containing 'methodology_details', 'results_data', and 'keywords'.",
    agent=notebook_parser_agent
)

# 2.2. Tâche de Rédaction (Coéquipier: Méthodologie)
task_write_methodology = Task(
    description=(
        "Write the complete 'Methodology' section of the research paper based on the details provided "
        "by the notebook parser. The tone must be formal and technical."
    ),
    expected_output="The full 'Methodology' section of the paper, in markdown format.",
    agent=methodology_writer_agent,
    context=[task_parse_notebook] # Dépend de l'extraction des données
)

# 2.3. Tâche de Rédaction (Coéquipier: Résultats)
task_write_results = Task(
    description=(
        "Write the 'Results and Discussion' section, presenting the quantitative data "
        "and discussing the findings in relation to the methodology. "
        "Reference all necessary tables and figures (even if they are conceptual at this stage)."
    ),
    expected_output="The full 'Results and Discussion' section of the paper, in markdown format.",
    agent=results_writer_agent,
    context=[task_parse_notebook] # Dépend de l'extraction des données
)

# 2.4. Tâche d'Amélioration (Coéquipier: Littéraire)
task_refine_literature = Task(
    description=(
        "Review the preliminary draft (Methodology and Results) for flow, clarity, and grammatical errors. "
        "Ensure the language is academic and the arguments are logically connected. "
    ),
    expected_output="The combined and polished 'Methodology' and 'Results' sections, ready for final critique.",
    agent=literary_agent,
    context=[task_write_methodology, task_write_results] # Dépend des deux sections écrites
)

# 2.5. Tâche de Critique (VOTRE TÂCHE)
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
    context=[task_refine_literature] # Doit s'exécuter sur le texte révisé
)

# 2.6. Tâche Finale (Coéquipier: Formatteur)
task_format_paper = Task(
    description=(
        "Take all sections (Methodology, Results, and Illustration Suggestions) and compile them "
        "into the final, APA-style research paper format, ready for submission. "
        "Place the Illustration Suggestions in an 'Appendix' or a 'Visual Requirements' section."
    ),
    expected_output="The final, complete research paper document, including all sections and visual notes.",
    agent=formatter_agent,
    context=[task_refine_literature, task_critique_visuals] # Dépend du texte + de votre critique
)


# --- 3. Assemble the Crew (Orchestration Finale) ---
papergen_crew = Crew(
    agents=[
        notebook_parser_agent,
        methodology_writer_agent,
        results_writer_agent,
        literary_agent,
        illustration_critic_agent,  # <-- VOTRE AGENT
        formatter_agent
    ],
    tasks=[
        task_parse_notebook,
        task_write_methodology,
        task_write_results,
        task_refine_literature,
        task_critique_visuals,  # <-- VOTRE TÂCHE est placée logiquement après la rédaction
        task_format_paper       # <-- Tâche finale
    ],
    process=Process.sequential,
    verbose=2
)

# --- 4. Create the API function ---
def run_papergen_crew(notebook_content: str, template: str):
    """
    Lance le Crew PaperGen pour générer le papier de recherche à partir du contenu d'un notebook.
    """
    inputs = {
        'notebook_content': notebook_content,
        'template': template # Si le formateur utilise un modèle de formatage
    }

    print("🚀 Démarrage du Crew PaperGen...")
    # Le kickoff doit idéalement prendre les inputs nécessaires
    result = papergen_crew.kickoff(inputs=inputs)
    
    print("--- 🏁 EXÉCUTION DU CREW TERMINÉE ---")
    return result