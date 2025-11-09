"""
Test script to diagnose formatter issues
"""
import sys
import os
import traceback

# Add the backend directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def test_formatter_components():
    """Test each formatter component to identify the issue"""
    
    print("🧪 Testing Formatter Components")
    print("=" * 50)
    
    try:
        # Test 1: Import orchestrator
        print("1. Testing orchestrator import...")
        from backend.app.agents.papergen.orchestrator_papergen import PapergenOrchestrator
        print("✅ Orchestrator imported successfully")
        
        # Test 2: Create orchestrator instance
        print("2. Testing orchestrator creation...")
        orchestrator = PapergenOrchestrator()
        print("✅ Orchestrator created successfully")
        
        # Test 3: Test the new figure embedding methods
        print("3. Testing figure embedding methods...")
        
        # Test _embed_figures_in_markdown
        test_markdown = "## Introduction\nThis is a test.\n\n## Results\nSome results here."
        test_illustrations = [
            {"file_path": "test.png", "caption": "Test Figure 1"},
            {"file_path": "test2.png", "caption": "Test Figure 2"}
        ]
        
        enhanced_markdown = orchestrator._embed_figures_in_markdown(test_markdown, test_illustrations)
        print(f"✅ _embed_figures_in_markdown works - output length: {len(enhanced_markdown)}")
        
        # Test 4: Test markdown to LaTeX conversion
        print("4. Testing markdown to LaTeX conversion...")
        
        # Create a mock state
        mock_state = {
            "illustrations": test_illustrations,
            "current_stage": "test",
            "progress": 50.0
        }
        
        latex_content = orchestrator._convert_markdown_to_latex(enhanced_markdown, mock_state)
        print(f"✅ _convert_markdown_to_latex works - output length: {len(latex_content)}")
        
        # Test 5: Check if LaTeX contains figures
        figure_count = latex_content.count('\\includegraphics')
        print(f"📊 LaTeX contains {figure_count} figures")
        
        if figure_count > 0:
            print("🎉 SUCCESS: Figures are being included in LaTeX!")
        else:
            print("⚠️  WARNING: No figures found in LaTeX output")
            # Print a sample to debug
            print("Sample LaTeX output:")
            print(latex_content[:500] + "..." if len(latex_content) > 500 else latex_content)
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        print("Full traceback:")
        traceback.print_exc()
        return False

def test_agent_tools():
    """Test the agent tools to see if they're working"""
    
    print("\n🔧 Testing Agent Tools")
    print("=" * 50)
    
    try:
        from backend.app.agents.agent_tools import (
            write_methodology,
            write_results, 
            synthesize_literary_sections,
            generate_illustrations,
            format_complete_paper
        )
        
        print("✅ All agent tools imported successfully")
        
        # Test format_complete_paper with mock data
        test_sections = {
            "abstract": "Test abstract",
            "introduction": "Test introduction", 
            "methodology_draft": "Test methodology",
            "results_draft": "Test results",
            "conclusion": "Test conclusion",
            "illustrations": [{"file_path": "test.png", "caption": "Test"}],
            "template": "Standard template"
        }
        
        sections_json = json.dumps(test_sections)
        result = format_complete_paper.invoke({"sections_json": sections_json})
        
        print(f"✅ format_complete_paper works - output length: {len(result)}")
        return True
        
    except Exception as e:
        print(f"❌ Agent tools test failed: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 Diagnosing PraxisAI Formatter Issues")
    print("=" * 50)
    
    import json
    
    # Test formatter components
    formatter_ok = test_formatter_components()
    
    # Test agent tools
    tools_ok = test_agent_tools()
    
    print("\n" + "=" * 50)
    print("📋 DIAGNOSIS SUMMARY:")
    print(f"   Formatter Components: {'✅ PASS' if formatter_ok else '❌ FAIL'}")
    print(f"   Agent Tools: {'✅ PASS' if tools_ok else '❌ FAIL'}")
    
    if formatter_ok and tools_ok:
        print("\n🎉 All components working! The issue might be elsewhere.")
        print("💡 Suggestions:")
        print("   - Check network connectivity for API calls")
        print("   - Verify API keys are correctly set")
        print("   - Check if the issue is in the frontend or backend communication")
    else:
        print("\n🚨 Issues found that need to be fixed!")
    
    exit(0 if (formatter_ok and tools_ok) else 1)