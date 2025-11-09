"""
Literary Agent for Research Paper Generation
Writes high-level narrative sections (Abstract, Introduction, Conclusion) by synthesizing information.
"""

import json
import re
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from backend.app.core.config import GROQ_API_KEY
except ImportError:
    # Fallback: try direct import
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from core.config import GROQ_API_KEY

class LiteraryOutputParser:
    """Custom parser for Literary Agent output."""
    
    def parse(self, text: str) -> Dict[str, str]:
        """Parse the agent output into structured sections."""
        sections = {}
        current_section = None
        current_content = []
        
        lines = text.split('\n')
        for line in lines:
            if line.strip().startswith('**Section:'):
                # Save previous section if exists
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                # Extract new section name
                section_name = line.strip().replace('**Section:', '').replace('**', '').strip()
                current_section = section_name.lower()
                current_content = []
            elif current_section:
                current_content.append(line)
        
        # Save the last section
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections

class LiteraryAgent:
    """
    Literary Agent that writes high-level narrative sections for research papers.
    
    Role: Writes Abstract, Introduction, and Conclusion by synthesizing all other parts.
    """
    
    def __init__(self):
        # Initialize ChatGroq with environment variable
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY environment variable is required")
        
        # Set environment variable for ChatGroq
        os.environ["GROQ_API_KEY"] = GROQ_API_KEY
        
        # Try multiple models in order of preference
        models_to_try = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant", 
            "llama3-8b-8192",
            "gemma2-9b-it",
            "mixtral-8x7b-32768"
        ]
        
        llm_instance = None
        for model in models_to_try:
            try:
                llm_instance = ChatGroq(
                    model=model,
                    temperature=0.3,
                    max_tokens=2000
                )
                print(f"✓ Successfully initialized with model: {model}")
                break
            except Exception as e:
                print(f"⚠ Failed to initialize model {model}: {str(e)[:100]}...")
                continue
        
        if not llm_instance:
            raise ValueError("Failed to initialize any supported ChatGroq model")
        
        self.llm = llm_instance
        self.parser = LiteraryOutputParser()
    
    def _analyze_content(self, content: str, content_type: str) -> Dict[str, Any]:
        """Analyze content to extract key information."""
        analysis = {
            "word_count": len(content.split()),
            "key_terms": [],
            "metrics": [],
            "content_type": content_type
        }
        
        if content_type == "methodology":
            # Extract methodology terms
            key_terms = ["neural network", "cnn", "model", "architecture", "algorithm", 
                        "training", "optimization", "framework", "tensorflow", "pytorch", 
                        "keras", "machine learning", "deep learning", "classification"]
            analysis["key_terms"] = [term for term in key_terms if term.lower() in content.lower()]
        
        elif content_type == "results":
            # Extract performance metrics
            metrics_patterns = [
                r'(\d+\.?\d*)%?\s*(?:accuracy|acc)',
                r'(\d+\.?\d*)\s*(?:loss|error)',
                r'(\d+\.?\d*)\s*(?:f1|precision|recall)',
                r'(\d+\.?\d*)\s*(?:epochs?|iterations?)'
            ]
            
            for pattern in metrics_patterns:
                matches = re.findall(pattern, content.lower())
                analysis["metrics"].extend(matches)
        
        return analysis
    
    def write_abstract(self, synthesis_data: Dict[str, Any]) -> str:
        """
        Write an abstract section based on synthesis data.
        
        Args:
            synthesis_data (Dict): Contains method_draft, results_draft
            
        Returns:
            str: Formatted abstract section
        """
        # Analyze the input data
        method_analysis = self._analyze_content(synthesis_data.get('method_draft', ''), 'methodology')
        results_analysis = self._analyze_content(synthesis_data.get('results_draft', ''), 'results')
        
        system_prompt = """You are an expert academic writer specializing in research paper abstracts. 
        Write concise, formal abstracts that follow academic standards. Use third person perspective 
        and past tense for research actions, present tense for general facts."""
        
        user_prompt = f"""
        Write an ABSTRACT section (150-250 words) using the following information:
        
        Methodology Draft: {synthesis_data.get('method_draft', '')}
        Key Technical Terms Found: {', '.join(method_analysis['key_terms'][:5])}
        
        Results Draft: {synthesis_data.get('results_draft', '')}
        Key Metrics Found: {', '.join(results_analysis['metrics'][:3])}
        
        Create a structured abstract that includes:
        1. Brief problem statement and objective
        2. Summary of methodology approach  
        3. Key quantitative results
        4. Main conclusion
        
        Format your response EXACTLY as:
        **Section: Abstract**
        [Your abstract content here - no additional formatting]
        """
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            result = str(response.content)
            
            # Ensure proper formatting
            if not result.startswith("**Section: Abstract**"):
                result = f"**Section: Abstract**\n{result}"
                
            return result
                
        except Exception as e:
            return f"**Section: Abstract**\nError generating abstract: {str(e)}"
    
    def write_introduction(self, synthesis_data: Dict[str, Any]) -> str:
        """
        Write an introduction section based on synthesis data.
        
        Args:
            synthesis_data (Dict): Contains method_draft, results_draft
            
        Returns:
            str: Formatted introduction section
        """
        # Analyze the input data
        method_analysis = self._analyze_content(synthesis_data.get('method_draft', ''), 'methodology')
        results_analysis = self._analyze_content(synthesis_data.get('results_draft', ''), 'results')
        
        system_prompt = """You are an expert academic writer specializing in research paper introductions. 
        Write engaging, well-structured introductions that establish context and motivation. Use formal 
        academic tone and logical progression from general to specific."""
        
        user_prompt = f"""
        Write an INTRODUCTION section (300-500 words) using the following information:
        
        Methodology Draft: {synthesis_data.get('method_draft', '')}
        Key Technical Approaches: {', '.join(method_analysis['key_terms'][:5])}
        
        Results Draft: {synthesis_data.get('results_draft', '')}
        Key Performance Indicators: {', '.join(results_analysis['metrics'][:3])}
        
        Create a structured introduction that includes:
        1. Background and context of the problem domain
        2. Brief literature review context
        3. Problem statement and research gap identification
        4. Research objectives and key contributions
        5. Brief paper organization overview
        
        Format your response EXACTLY as:
        **Section: Introduction**
        [Your introduction content here - no additional formatting]
        """
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            result = str(response.content)
            
            # Ensure proper formatting
            if not result.startswith("**Section: Introduction**"):
                result = f"**Section: Introduction**\n{result}"
                
            return result
                
        except Exception as e:
            return f"**Section: Introduction**\nError generating introduction: {str(e)}"
    
    def write_conclusion(self, synthesis_data: Dict[str, Any]) -> str:
        """
        Write a conclusion section based on synthesis data.
        
        Args:
            synthesis_data (Dict): Contains method_draft, results_draft
            
        Returns:
            str: Formatted conclusion section
        """
        # Analyze the input data
        method_analysis = self._analyze_content(synthesis_data.get('method_draft', ''), 'methodology')
        results_analysis = self._analyze_content(synthesis_data.get('results_draft', ''), 'results')
        
        system_prompt = """You are an expert academic writer specializing in research paper conclusions. 
        Write impactful conclusions that summarize findings, discuss implications, and suggest future 
        directions. Use formal academic tone and forward-looking perspective."""
        
        user_prompt = f"""
        Write a CONCLUSION section (200-400 words) using the following information:
        
        Methodology Draft: {synthesis_data.get('method_draft', '')}
        Technical Approaches Used: {', '.join(method_analysis['key_terms'][:5])}
        
        Results Draft: {synthesis_data.get('results_draft', '')}
        Key Achievements: {', '.join(results_analysis['metrics'][:3])}
        
        Create a structured conclusion that includes:
        1. Summary of key findings and achievements
        2. Significance and implications of results
        3. Limitations and constraints acknowledged
        4. Future work and research directions
        5. Final impactful concluding statement
        
        Format your response EXACTLY as:
        **Section: Conclusion**
        [Your conclusion content here - no additional formatting]
        """
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            result = str(response.content)
            
            # Ensure proper formatting
            if not result.startswith("**Section: Conclusion**"):
                result = f"**Section: Conclusion**\n{result}"
                
            return result
                
        except Exception as e:
            return f"**Section: Conclusion**\nError generating conclusion: {str(e)}"
    
    def process_task(self, task_description: str, synthesis_data: Dict[str, Any]) -> str:
        """
        Process a task from the orchestrator.
        
        Args:
            task_description (str): Task description from orchestrator
            synthesis_data (Dict): Data containing method_draft, results_draft
            
        Returns:
            str: Generated content based on the task
        """
        task_lower = task_description.lower()
        
        if "full report" in task_lower or "complete report" in task_lower:
            return self.generate_full_report(synthesis_data)
        elif "abstract" in task_lower and "conclusion" in task_lower:
            # Write both abstract and conclusion as per the example requirement
            abstract = self.write_abstract(synthesis_data)
            conclusion = self.write_conclusion(synthesis_data)
            return f"{abstract}\n\n{conclusion}"
        elif "abstract" in task_lower:
            return self.write_abstract(synthesis_data)
        elif "introduction" in task_lower:
            return self.write_introduction(synthesis_data)
        elif "conclusion" in task_lower:
            return self.write_conclusion(synthesis_data)
        else:
            # Default: generate full report
            return self.generate_full_report(synthesis_data)
    
    def synthesize_all_sections(self, synthesis_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate all narrative sections for a research paper.
        
        Args:
            synthesis_data (Dict): Data containing method_draft, results_draft
            
        Returns:
            Dict[str, str]: Dictionary with keys 'abstract', 'introduction', 'conclusion'
        """
        return {
            'abstract': self.write_abstract(synthesis_data),
            'introduction': self.write_introduction(synthesis_data),
            'conclusion': self.write_conclusion(synthesis_data)
        }
    
    def generate_full_report(self, synthesis_data: Dict[str, Any]) -> str:
        """
        Generate a complete research paper report including all sections.
        
        Args:
            synthesis_data (Dict): Data containing method_draft, results_draft
            
        Returns:
            str: Complete formatted research paper with all sections
        """
        try:
            # Generate narrative sections
            abstract = self.write_abstract(synthesis_data)
            introduction = self.write_introduction(synthesis_data)
            conclusion = self.write_conclusion(synthesis_data)
            
            # Format methodology section
            methodology_content = synthesis_data.get('method_draft', '')
            methodology_section = f"**Section: Methodology**\n{methodology_content}"
            
            # Format results section
            results_content = synthesis_data.get('results_draft', '')
            results_section = f"**Section: Results**\n{results_content}"
            
            # Combine all sections into a complete report
            full_report = f"""# Research Paper Report

{abstract}

{introduction}

{methodology_section}

{results_section}

{conclusion}

---
*Report generated by Literary Agent*"""
            
            return full_report
            
        except Exception as e:
            return f"Error generating full report: {str(e)}"


# Convenience function for direct usage
def create_literary_agent() -> LiteraryAgent:
    """Create and return a Literary Agent instance."""
    return LiteraryAgent()


# Tool function for use with agent_tools.py
def literary_agent_tool(task_description: str, synthesis_data_json: str) -> str:
    """
    Tool function for the Literary Agent that can be used by other agents.
    
    Args:
        task_description (str): Task description (e.g., "Generate full report", "Write ABSTRACT")
        synthesis_data_json (str): JSON string containing synthesis data with keys:
                                 - method_draft: Methodology section content
                                 - results_draft: Results section content
        
    Returns:
        str: Generated literary content or complete research paper report
    """
    try:
        synthesis_data = json.loads(synthesis_data_json)
        agent = create_literary_agent()
        return agent.process_task(task_description, synthesis_data)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON format - {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    # Run example
    print("Literary Agent Test:")
    print("=" * 50)
    
    try:
        # Create agent
        agent = LiteraryAgent()
        print("✓ Literary Agent created successfully")
        
        # Test synthesis data (removed user_notes)
        synthesis_data = {
            "method_draft": "We employed a sequential Convolutional Neural Network (CNN) architecture built using the TensorFlow Keras API. The model's architecture begins with a 2D Convolutional layer composed of 32 filters with a 3x3 kernel size, followed by ReLU activation and MaxPooling layers. The network includes additional convolutional layers with increasing filter sizes (64, 128) to capture hierarchical features. The final layers consist of a Flatten layer, Dense layers with dropout regularization, and a softmax output layer for 10-class classification.",
            "results_draft": "The model was trained for 20 epochs using the Adam optimizer with a learning rate of 0.001. Training accuracy reached 99.2% while validation accuracy stabilized at 98.7%. The final test accuracy achieved was 91.5% on the held-out MNIST test dataset of 10,000 images. The confusion matrix revealed strong performance across all digit classes, with minor confusion between digits 4 and 9. Training completed in approximately 15 minutes on GPU hardware."
        }
        
        print("✓ Test data prepared (without user_notes)")
        
        # Test full report generation (only if GROQ_API_KEY is available)
        if GROQ_API_KEY:
            print("✓ GROQ_API_KEY found, testing full report generation...")
            full_report = agent.generate_full_report(synthesis_data)
            print("✓ Full report generated successfully")
            print("\nGenerated Full Report Preview:")
            print(full_report[:400] + "..." if len(full_report) > 400 else full_report)
        else:
            print("⚠ GROQ_API_KEY not found - skipping API test")
        
        print("\nLiterary Agent is ready for use!")
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
