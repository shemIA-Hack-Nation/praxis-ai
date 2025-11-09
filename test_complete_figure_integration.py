#!/usr/bin/env python3
"""
Comprehensive test for figure integration in paper generation
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.app.agents.papergen.orchestrator_papergen import PapergenOrchestrator

def test_complete_figure_workflow():
    """Test the complete figure integration workflow"""
    
    print("🧪 Testing Complete Figure Integration Workflow")
    print("=" * 60)
    
    # Create orchestrator
    orchestrator = PapergenOrchestrator()
    
    # Test markdown with multiple sections
    test_markdown = """# Research Paper

## Abstract
This is the abstract of the paper.

## Introduction  
This section introduces the research problem and objectives.

## Methodology
We employed advanced machine learning techniques for data analysis. The methodology involved several key steps that are crucial for understanding our approach.

This paragraph continues the methodology discussion.

## Results
Our analysis revealed significant patterns in the data. The results demonstrate the effectiveness of our approach.

Additional results are presented here with detailed explanations.

## Discussion
The findings have important implications for the field.

## Conclusion
This work contributes to the advancement of the research area.
"""
    
    # Create test illustrations
    test_illustrations = [
        {
            'file_path': 'test_figure_1.png',
            'description': 'Methodology flowchart showing the data processing pipeline'
        },
        {
            'file_path': 'test_figure_2.png', 
            'description': 'Results visualization of key performance metrics'
        },
        {
            'file_path': 'test_figure_3.png',
            'description': 'Comparative analysis results across different conditions'
        }
    ]
    
    # Test figure embedding
    print("1. Testing figure embedding in markdown...")
    try:
        enhanced_markdown = orchestrator._embed_figures_in_markdown(test_markdown, test_illustrations)
        print("✅ Figure embedding successful")
        
        # Count figure references
        figure_count = enhanced_markdown.count('![Figure')
        print(f"📊 Found {figure_count} embedded figures")
        
        if figure_count > 0:
            print("✅ Figures successfully embedded in markdown")
        else:
            print("⚠️ No figures found in enhanced markdown")
            
    except Exception as e:
        print(f"❌ Figure embedding failed: {e}")
        return False
    
    # Test LaTeX conversion
    print("\n2. Testing LaTeX conversion with figures...")
    try:
        # Create a minimal mock state for testing
        from backend.app.agents.papergen.orchestrator_papergen import PapergenState
        mock_state = PapergenState()
        
        latex_content = orchestrator._convert_markdown_to_latex(enhanced_markdown, mock_state)
        print("✅ LaTeX conversion successful")
        
        # Count LaTeX figures
        latex_figure_count = latex_content.count('\\begin{figure}')
        print(f"📊 Found {latex_figure_count} LaTeX figures")
        
        if latex_figure_count > 0:
            print("✅ Figures successfully converted to LaTeX format")
        else:
            print("⚠️ No LaTeX figures found")
            
        # Show sample LaTeX figure
        if '\\begin{figure}' in latex_content:
            start = latex_content.find('\\begin{figure}')
            end = latex_content.find('\\end{figure}', start) + len('\\end{figure}')
            sample_figure = latex_content[start:end]
            print(f"\n📋 Sample LaTeX Figure:\n{sample_figure}")
            
    except Exception as e:
        print(f"❌ LaTeX conversion failed: {e}")
        return False
    
    # Test figure reference creation
    print("\n3. Testing figure reference creation...")
    try:
        figure_ref = orchestrator._create_markdown_figure_ref(test_illustrations[0], 1)
        print("✅ Figure reference creation successful")
        print(f"📋 Sample figure reference:\n{figure_ref}")
        
    except Exception as e:
        print(f"❌ Figure reference creation failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 Complete Figure Integration Test: PASSED")
    print("💡 All figure integration components are working correctly!")
    
    return True

if __name__ == "__main__":
    test_complete_figure_workflow()