import os
import re
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Configuration Import ---
try:
    from backend.app.core.config import GROQ_API_KEY
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    
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
                temperature=0.1,  # Low temperature for consistent formatting
                max_tokens=3000,
                stop_sequences=None
            )
            print(f"✓ FormatterAgent initialized with Groq model: {model}")
            break
        except Exception as e:
            print(f"⚠ Failed to initialize Groq model {model}: {str(e)[:100]}...")
            continue
    
    if not llm:
        raise ValueError("❌ Failed to initialize any supported Groq model for FormatterAgent")
        
except Exception as e:
    print(f"Error (agent_formatter): {e}")
    llm = None

# --- Agent: FormatterAgent (Paper Assembly and Formatting) ---
def format_paper(sections_data: Dict[str, Any], template: Optional[str] = None) -> str:
    """
    Format and assemble a complete research paper from individual sections.
    
    Args:
        sections_data (Dict): Dictionary containing:
            - methodology_draft: str
            - results_draft: str
            - abstract: str
            - introduction: str
            - conclusion: str
            - illustrations: List[Dict] (optional)
            - template: str (optional)
            
    Returns:
        str: Complete formatted research paper
    """
    if llm is None:
        return "Error: LLM (Groq) is not initialized for FormatterAgent."
        
    print("--- Calling FormatterAgent (via Groq) ---")
    
    try:
        # Extract sections
        methodology = sections_data.get("methodology_draft", "")
        results = sections_data.get("results_draft", "")
        abstract = sections_data.get("abstract", "")
        introduction = sections_data.get("introduction", "")
        conclusion = sections_data.get("conclusion", "")
        illustrations = sections_data.get("illustrations", [])
        
        # Use provided template or create default
        if template:
            paper_template = template
        else:
            paper_template = _get_default_template()
        
        # Prepare sections for formatting
        sections_content = {
            "abstract": abstract,
            "introduction": introduction, 
            "methodology": methodology,
            "results": results,
            "conclusion": conclusion
        }
        
        # Create illustration references
        illustration_refs = _create_illustration_references(illustrations)
        
        # Format the complete paper
        formatted_paper = _assemble_paper_with_llm(
            sections_content, 
            paper_template, 
            illustration_refs
        )
        
        return formatted_paper
        
    except Exception as e:
        return f"Error during paper formatting: {str(e)}"

def _get_default_template() -> str:
    """Get default research paper template"""
    return """# {TITLE}

## Abstract
{ABSTRACT}

## 1. Introduction
{INTRODUCTION}

## 2. Methodology
{METHODOLOGY}

## 3. Results
{RESULTS}

## 4. Conclusion
{CONCLUSION}

## References
{REFERENCES}

## Figures
{FIGURES}
"""

def _create_illustration_references(illustrations: List[Dict]) -> str:
    """Create formatted illustration references"""
    if not illustrations:
        return ""
    
    refs = []
    for i, ill in enumerate(illustrations, 1):
        caption = ill.get("caption", f"Figure {i}")
        path = ill.get("path", "")
        description = ill.get("description", "")
        
        ref = f"**Figure {i}:** {caption}"
        if description:
            ref += f"\n*{description}*"
        if path:
            ref += f"\n*File: {path}*"
        refs.append(ref)
    
    return "\n\n".join(refs)

def _assemble_paper_with_llm(sections: Dict[str, str], template: str, illustrations: str) -> str:
    """Use LLM to intelligently assemble and format the paper"""
    
    # Create formatting prompt
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a professional academic paper formatter. Your task is to assemble "
            "individual sections into a coherent, well-formatted research paper. "
            "Ensure proper academic structure, consistent formatting, and smooth transitions "
            "between sections. Add appropriate section numbering and formatting."
        )),
        ("user", """
        Assemble the following sections into a complete, well-formatted research paper:

        TEMPLATE STRUCTURE:
        {template}

        SECTIONS TO INTEGRATE:
        
        Abstract Section:
        {abstract}
        
        Introduction Section:
        {introduction}
        
        Methodology Section:
        {methodology}
        
        Results Section:
        {results}
        
        Conclusion Section:
        {conclusion}
        
        Available Figures/Illustrations:
        {illustrations}

        Instructions:
        1. Create a coherent paper following academic standards
        2. Add an appropriate title based on the content
        3. Ensure smooth transitions between sections
        4. Include proper section numbering (1, 2, 3, etc.)
        5. Integrate figure references where appropriate
        6. Add a basic References section placeholder
        7. Format using markdown syntax
        8. Ensure consistency in tone and style throughout
        
        Output the complete formatted paper:
        """)
    ])
    
    # Execute formatting
    if llm is None:
        return "Error: LLM not initialized for paper assembly"
        
    chain = prompt_template | llm | StrOutputParser()
    
    return chain.invoke({
        "template": template,
        "abstract": sections.get("abstract", ""),
        "introduction": sections.get("introduction", ""),
        "methodology": sections.get("methodology", ""),
        "results": sections.get("results", ""),
        "conclusion": sections.get("conclusion", ""),
        "illustrations": illustrations
    })

def clean_and_validate_paper(paper_content: str) -> Dict[str, Any]:
    """
    Clean and validate the formatted paper content.
    
    Args:
        paper_content (str): Raw formatted paper content
        
    Returns:
        Dict containing:
            - cleaned_content: str
            - validation_issues: List[str]
            - word_count: int
            - section_count: int
    """
    
    # Clean the content
    cleaned = paper_content.strip()
    
    # Remove excessive whitespace
    cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)
    
    # Validation checks
    issues = []
    
    # Check for required sections
    required_sections = ['abstract', 'introduction', 'methodology', 'results', 'conclusion']
    for section in required_sections:
        if section.lower() not in cleaned.lower():
            issues.append(f"Missing or unclear {section} section")
    
    # Check for minimum content length
    if len(cleaned) < 500:
        issues.append("Paper content appears too short (< 500 characters)")
    
    # Count sections (looking for # headers)
    section_count = len(re.findall(r'^#+\s+', cleaned, re.MULTILINE))
    
    # Estimate word count
    word_count = len(cleaned.split())
    
    return {
        "cleaned_content": cleaned,
        "validation_issues": issues,
        "word_count": word_count,
        "section_count": section_count
    }

def generate_paper_metadata(sections_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate metadata for the assembled paper.
    
    Args:
        sections_data: Dictionary containing all paper sections
        
    Returns:
        Dictionary containing paper metadata
    """
    
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "generator": "PraxisAI FormatterAgent",
        "sections_included": [],
        "total_characters": 0,
        "estimated_reading_time": 0,
        "illustrations_count": 0
    }
    
    # Analyze included sections
    for section_name, content in sections_data.items():
        if isinstance(content, str) and content.strip():
            metadata["sections_included"].append(section_name)
            metadata["total_characters"] += len(content)
        elif section_name == "illustrations" and isinstance(content, list):
            metadata["illustrations_count"] = len(content)
    
    # Estimate reading time (average 200 words per minute)
    word_count = metadata["total_characters"] / 5  # Rough estimate
    metadata["estimated_reading_time"] = max(1, int(word_count / 200))
    
    return metadata
