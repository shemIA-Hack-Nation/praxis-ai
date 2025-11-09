#!/usr/bin/env python3
"""
Research Context API - Returns structured JSON output
Input: {"keywords": ["CNN", "Image Classification", "Survey"], "references": ["LeCun-1998"]}
Output: {"surveys": [...], "cited_by": [...]}
"""

import json
import os
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

def search_academic_papers(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Search for academic papers using DuckDuckGo.
    
    Returns:
        List of dictionaries with title, url, snippet
    """
    try:
        from ddgs import DDGS
        
        ddgs = DDGS()
        results = ddgs.text(query, max_results=max_results)
        
        if not results:
            return []

        formatted_results = []
        for r in results:
            title = r.get('title', 'N/A')
            url = r.get('href', 'N/A')
            snippet = r.get('body', r.get('snippet', 'N/A'))
            
            formatted_results.append({
                'title': title,
                'url': url,
                'snippet': snippet
            })
        
        return formatted_results

    except ImportError:
        print("Warning: DDGS package not installed. Using simulated results.")
        return [
            {
                'title': 'Example Survey Paper',
                'url': 'https://example.com/survey',
                'snippet': 'This is a simulated search result for testing purposes.'
            }
        ]
    except Exception as e:
        print(f"Search error: {e}")
        return []

def get_llm():
    """Get configured LLM instance."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    model_name = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")
    
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.2,
        google_api_key=api_key
    )

def extract_structured_papers(search_results: List[Dict], keywords: List[str], 
                            paper_type: str) -> List[Dict[str, str]]:
    """
    Use LLM to extract and structure paper information.
    
    Args:
        search_results: Raw search results
        keywords: Research keywords
        paper_type: "survey" or "cited_by"
    
    Returns:
        List of structured paper dictionaries
    """
    if not search_results:
        return []
    
    try:
        llm = get_llm()
        
        # Create a more specific prompt that forces JSON output
        prompt_template = ChatPromptTemplate.from_template("""
You are an expert academic researcher. Your task is to extract paper information and return it as valid JSON.

Keywords: {keywords}
Paper Type: {paper_type}

Search Results:
{search_results}

Task: Extract the most relevant papers and format them as JSON.

Rules:
1. Select papers most relevant to keywords: {keywords}
2. For {paper_type} papers, focus on comprehensive reviews and surveys
3. Create concise summaries (1-2 sentences max)
4. Use the exact URLs provided
5. Return EXACTLY this JSON format with no additional text:

[
  {
    "title": "Exact paper title from search results",
    "url": "Exact URL from search results", 
    "summary": "Brief 1-2 sentence summary of the paper's contribution"
  }
]

IMPORTANT: Return ONLY the JSON array. No explanations, no markdown, just the JSON.
""")
        
        # Format search results more clearly
        results_text = ""
        for i, r in enumerate(search_results[:5], 1):  # Limit to top 5 results
            results_text += f"\nResult {i}:\n"
            results_text += f"Title: {r['title']}\n"
            results_text += f"URL: {r['url']}\n"
            results_text += f"Snippet: {r['snippet'][:300]}...\n"
            results_text += "-" * 50 + "\n"
        
        # Get LLM response
        messages = prompt_template.format_messages(
            paper_type=paper_type,
            keywords=", ".join(keywords),
            search_results=results_text
        )
        
        response = llm.invoke(messages)
        response_text = response.content.strip()
        
        print(f"\n🤖 LLM Response for {paper_type}:")
        print(f"Raw response: {response_text[:200]}...")
        
        # Try to extract JSON from the response
        try:
            # Look for JSON array in the response
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = response_text[start_idx:end_idx]
                papers = json.loads(json_str)
                
                if isinstance(papers, list):
                    print(f"✅ Successfully parsed {len(papers)} papers for {paper_type}")
                    return papers[:3]  # Limit to top 3 papers
            
            # If JSON parsing fails, create structured output from search results
            print(f"⚠️ JSON parsing failed for {paper_type}, creating structured output from search results")
            return create_fallback_papers(search_results[:3], keywords, paper_type)
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error for {paper_type}: {e}")
            return create_fallback_papers(search_results[:3], keywords, paper_type)
            
    except Exception as e:
        print(f"❌ Error extracting {paper_type} papers: {e}")
        return create_fallback_papers(search_results[:3], keywords, paper_type)

def create_fallback_papers(search_results: List[Dict], keywords: List[str], 
                          paper_type: str) -> List[Dict[str, str]]:
    """
    Create structured papers from search results when LLM fails.
    """
    papers = []
    
    for result in search_results:
        title = result.get('title', 'Unknown Title')
        url = result.get('url', '#')
        snippet = result.get('snippet', 'No description available')
        
        # Create a basic summary
        if paper_type == "survey":
            summary = f"Survey paper on {', '.join(keywords)}. {snippet[:100]}..."
        else:
            summary = f"Research paper related to {', '.join(keywords)}. {snippet[:100]}..."
        
        papers.append({
            "title": title,
            "url": url,
            "summary": summary[:200] + "..." if len(summary) > 200 else summary
        })
    
    print(f"✅ Created {len(papers)} fallback papers for {paper_type}")
    return papers

def find_context_for(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function to find research context and return structured JSON.
    
    Args:
        input_data: {"keywords": [...], "references": [...]}
    
    Returns:
        {"surveys": [...], "cited_by": [...]}
    """
    keywords = input_data.get("keywords", [])
    references = input_data.get("references", [])
    
    print(f"🔍 Finding context for keywords: {keywords}")
    print(f"📚 References: {references}")
    
    # Search for survey papers with better queries
    print("🔍 Searching for survey papers...")
    survey_queries = [
        f"survey {' '.join(keywords)} comprehensive review",
        f"{' '.join(keywords)} survey paper state of the art",
        f"review {' '.join(keywords)} recent advances"
    ]
    
    all_survey_results = []
    for query in survey_queries:
        results = search_academic_papers(query, max_results=3)
        all_survey_results.extend(results)
        if len(all_survey_results) >= 6:
            break
    
    print(f"📊 Found {len(all_survey_results)} survey search results")
    
    # Search for papers citing the references
    print("🔍 Searching for citing papers...")
    citing_queries = [
        f"{' '.join(references)} {' '.join(keywords)} applications",
        f"{' '.join(keywords)} based on {' '.join(references)}",
        f"deep learning {' '.join(keywords)} {' '.join(references)}"
    ]
    
    all_citing_results = []
    for query in citing_queries:
        results = search_academic_papers(query, max_results=3)
        all_citing_results.extend(results)
        if len(all_citing_results) >= 6:
            break
    
    print(f"📊 Found {len(all_citing_results)} citing search results")
    
    # If still no results, ensure we have some data
    if not all_survey_results:
        print("⚠️ No survey results found, using fallback search")
        fallback_query = f"{' '.join(keywords)} research papers"
        all_survey_results = search_academic_papers(fallback_query, max_results=3)
    
    if not all_citing_results:
        print("⚠️ No citing results found, using fallback search")
        fallback_query = f"{' '.join(keywords)} recent work"
        all_citing_results = search_academic_papers(fallback_query, max_results=3)
    
    # Extract structured information using LLM
    print("🤖 Extracting structured paper information...")
    surveys = extract_structured_papers(all_survey_results, keywords, "survey")
    cited_by = extract_structured_papers(all_citing_results, keywords, "cited_by")
    
    # Ensure we have at least some results
    if not surveys and all_survey_results:
        print("🔧 Creating fallback survey papers")
        surveys = create_fallback_papers(all_survey_results[:2], keywords, "survey")
    
    if not cited_by and all_citing_results:
        print("🔧 Creating fallback citing papers")
        cited_by = create_fallback_papers(all_citing_results[:2], keywords, "cited_by")
    
    # Return structured JSON
    result = {
        "surveys": surveys,
        "cited_by": cited_by
    }
    
    print(f"✅ Final result: {len(surveys)} survey papers and {len(cited_by)} citing papers")
    return result

def main():
    """Main function for testing and demonstration."""
    
    # Example input - your exact format
    input_data = {
        "keywords": ["CNN", "Image Classification", "Survey"],
        "references": ["LeCun-1998"]
    }
    
    print("🚀 Research Context API")
    print("=" * 50)
    print(f"📝 Input: {json.dumps(input_data, indent=2)}")
    print()
    
    try:
        # Process the request
        result = find_context_for(input_data)
        
        # Output the JSON result
        print("📊 JSON Output:")
        print("=" * 50)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # Save to file
        with open('research_output.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print("\n✅ Results saved to 'research_output.json'")
        
        return result
        
    except Exception as e:
        error_result = {
            "surveys": [],
            "cited_by": [],
            "error": str(e)
        }
        print(f"❌ Error: {e}")
        print(json.dumps(error_result, indent=2))
        return error_result

# API-style function for direct usage
def process_research_request(keywords: List[str], references: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    API-style function for direct usage.
    
    Args:
        keywords: List of research keywords
        references: Optional list of reference papers
    
    Returns:
        {"surveys": [...], "cited_by": [...]}
    """
    input_data = {
        "keywords": keywords,
        "references": references or []
    }
    
    return find_context_for(input_data)

if __name__ == "__main__":
    main()