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
            print(f"✓ ResultsWriter initialized with Groq model: {model}")
            break
        except Exception as e:
            print(f"⚠ Failed to initialize Groq model {model}: {str(e)[:100]}...")
            continue
    
    if not llm:
        raise ValueError("❌ Failed to initialize any supported Groq model for ResultsWriter")
        
except Exception as e:
    print(f"Error (agent_results_writer): {e}")
    llm = None

load_dotenv()

# --- Agent 4: ResultsWriter (LLM Chain - Groq) ---
def results_writer(json_data: dict) -> str:
    """
    Writes the results section by analyzing both 'code' and 'output_text' cells
    from the parsed JSON data.
    """
    if llm is None:
        return "Error: LLM (Groq) is not initialized."
        
    print("--- Calling ResultsWriter (via Groq) ---")
    
    relevant_snippets = []
    
    # --- 1. Extract relevant context (Code + Outputs) ---
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


            # A. Capture relevant code (training or evaluation)
            if cell_type == "code":
                if "model.fit" in cell_content or ".fit(" in cell_content or \
                   "model.evaluate" in cell_content or ".evaluate(" in cell_content:
                    
                    relevant_snippets.append(f"Code executed:\n```python\n{cell_content}\n```")

            # B. Capture relevant text outputs
            elif cell_type == "output_text":
                # Filter out noise (e.g., <keras...History...>)
                if cell_content.startswith("<") or not cell_content:
                    continue 
                
                # Capture epoch logs (e.g., Epoch 5/5... val_accuracy: 0.7992)
                if "Epoch" in cell_content or "val_accuracy" in cell_content:
                    relevant_snippets.append(f"Training log (output_text):\n{cell_content}\n")
                
                # Capture evaluation logs (e.g., 92/92 ... accuracy: 0.81)
                elif "/" in cell_content and ("accuracy" in cell_content or "loss" in cell_content):
                     relevant_snippets.append(f"Evaluation log (output_text):\n{cell_content}\n")

    except Exception as e:
        return f"Error during internal JSON parsing: {e}"

    if not relevant_snippets:
        return "Results Section: No relevant code (fit/evaluate) or 'output_text' cells were found."

    # Combine all context into a single string
    combined_context = "\n---\n".join(relevant_snippets)

    # --- 2. Create a smarter prompt (IN ENGLISH) ---
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a technical research writer. Your task is to analyze a log of "
            "code cells and notebook outputs (logs) to write a **Results** section."
            "Ignore all escape characters (e.g., \\u001b[1m)."
        )),
        ("user", """
        Analyze the following context, which shows the executed code (e.g., `model.evaluate`) and the output logs (output_text) that followed.
        
        Notebook Context:
        {context}
        
        Write a clear and concise **Results** section.
        Please do the following:
        1.  Look at the training logs (those with "Epoch") to find the final validation accuracy (`val_accuracy`) and loss (`val_loss`) from the *last epoch*.
        2.  Look at the evaluation code (e.g., `.evaluate(test_data)`) and the logs that follow it (e.g., `92/92...`) to find the final test accuracy (`accuracy`) and loss (`loss`).
        3.  Synthesize these numbers into formal prose. Do not mention epoch numbers, only the final results.
        """)
    ])
    
    # --- 3. Execute the chain ---
    chain = prompt_template | llm | StrOutputParser()
    
    return chain.invoke({"context": combined_context})