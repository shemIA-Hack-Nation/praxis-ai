import json
import os
import sys
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# --- Configuration Import ---
# Use the same pattern as other agents in this project
try:
    from backend.app.core.config import GROQ_API_KEY
    from langchain_groq import ChatGroq
    
    # Initialize LLM with Groq
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY environment variable is required")
    
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
    
    # Try multiple models in order of preference
    models_to_try = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant", 
        "llama3-8b-8192",
        "gemma2-9b-it",
        "mixtral-8x7b-32768"
    ]
    
    llm = None
    for model in models_to_try:
        try:
            llm = ChatGroq(
                model=model,
                temperature=0.3,
                max_tokens=2000,
                stop_sequences=None
            )
            print(f"✓ MethodologyWriter initialized with Groq model: {model}")
            break
        except Exception as e:
            print(f"⚠ Failed to initialize Groq model {model}: {str(e)[:100]}...")
            continue
    
    if not llm:
        raise ValueError("❌ Failed to initialize any supported Groq model for MethodologyWriter")
        
except Exception as e:
    print(f"Error (agent_methodology_writer): {e}")
    llm = None

load_dotenv()

# --- Agent 3: MethodologyWriter (LLM Chain - Groq) ---
def methodology_writer(json_data: dict) -> str:
    """
    Writes the methodology section by analyzing 'code' cells to understand
    the technical approach and implementation details.
    """
    if llm is None:
        return "Error: LLM (Groq) is not initialized."
        
    print("--- Calling MethodologyWriter (via Groq) ---")
    
    relevant_snippets = []
    
    # --- 1. Extract relevant methodology context (Code Analysis) ---
    try:
        for cell in json_data.get("cells", []):
            cell_type = cell.get("type")
            cell_content_raw = cell.get("content") # Get raw content

            # --- START OF FIX ---
            # We need to ensure that 'cell_content' is a string
            
            cell_content = "" # Initialize as empty string
            
            if isinstance(cell_content_raw, str):
                cell_content = cell_content_raw.strip()
            elif isinstance(cell_content_raw, list):
                # If content is a list (e.g., code or markdown on multiple lines)
                cell_content = "\n".join(cell_content_raw).strip()
            elif isinstance(cell_content_raw, dict):
                # If content is a dict (e.g., rich output)
                # We try to extract the plain text version
                cell_content = cell_content_raw.get("text/plain", "").strip()
            # If it's None or something else, cell_content remains ""
            
            # --- END OF FIX ---

            # A. Capture methodology-relevant code
            if cell_type == "code":
                # Look for data preprocessing, model definition, architecture setup
                if any(keyword in cell_content.lower() for keyword in [
                    "import", "from", "load_data", "preprocess", "model", "sequential", 
                    "dense", "conv", "lstm", "dropout", "compile", "optimizer", 
                    "loss", "metric", "split", "train_test_split", "normalize",
                    "standardize", "reshape", "transform", "pipeline"
                ]):
                    relevant_snippets.append(f"Implementation code:\n```python\n{cell_content}\n```")

            # B. Capture methodology-relevant markdown explanations
            elif cell_type == "markdown":
                # Look for methodology descriptions, approach explanations
                if any(keyword in cell_content.lower() for keyword in [
                    "method", "approach", "algorithm", "model", "architecture",
                    "preprocessing", "feature", "parameter", "hyperparameter",
                    "training", "validation", "cross-validation", "technique"
                ]):
                    relevant_snippets.append(f"Methodology description:\n{cell_content}\n")

    except Exception as e:
        return f"Error during internal JSON parsing: {e}"

    if not relevant_snippets:
        return "Methodology Section: No relevant methodology code or descriptions were found."

    # Combine all context into a single string
    combined_context = "\n---\n".join(relevant_snippets)

    # --- 2. Create methodology-focused prompt ---
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a technical research writer specializing in methodology sections. "
            "Your task is to analyze code implementations and descriptions to write a "
            "comprehensive **Methodology** section for a research paper."
        )),
        ("user", """
        Analyze the following context, which shows the implementation code and methodology descriptions from a research notebook.
        
        Notebook Context:
        {context}
        
        Write a clear and comprehensive **Methodology** section.
        Please do the following:
        1. Describe the overall approach and technical framework used
        2. Explain the data preprocessing steps if present
        3. Detail the model architecture and key components
        4. Describe the training procedure and hyperparameters
        5. Mention evaluation methodology and validation approach
        6. Use technical language appropriate for an academic paper
        7. Focus on the "how" rather than the "what" - explain the process and reasoning
        
        Structure the output as a cohesive methodology section without subsection headers.
        """)
    ])
    
    # --- 3. Execute the chain ---
    chain = prompt_template | llm | StrOutputParser()
    
    return chain.invoke({"context": combined_context})
