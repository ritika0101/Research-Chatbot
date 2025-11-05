"""
Step 1: Planner (Gemma via Groq) — Decompose query into sub-questions
"""

import os, json, re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()

def planner(query, depth="normal", model="gemma2-9b-it"):
    """
    Uses Gemma model through Groq API to decompose a query into sub-questions
    
    Args:
        query (str): The high-level query to decompose
        depth (str): Research depth level ("shallow", "normal", "deep")
        model (str): Groq model to use (gemma2-9b-it, gemma-7b-it)
    """
    groq_api_key = "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"

    # Initialize ChatGroq with Gemma model
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model,
        temperature=0.1,  # Low temperature for consistent JSON output
        max_tokens=1024
    )

    # Depth-specific instructions
    depth_instructions = {
        "shallow": "Produce only 3–4 very broad sub-questions that summarize the main themes. Keep them high-level, without going into detail.",
        "normal": "Produce 6–7 balanced sub-questions covering policies, impacts, stakeholders, and logistics. Provide a moderate level of detail.",
        "deep": "Produce 8–12 fine-grained sub-questions, exploring detailed policies, sector-specific impacts, edge cases, and long-term implications."
    }

    depth_note = depth_instructions.get(depth, depth_instructions["normal"])

    # Construct prompt
    prompt = f"""
You are ResearchPlanner. Decompose this high-level query into sub-questions.

Query: "{query}"

Depth setting: {depth} → {depth_note}

Output ONLY JSON in this schema:
{{ "query":"{query}", "depth":"{depth}", "subquestions":[
  {{ "id":"Q1","category":"scope|stakeholders|causes|impacts|policy|logistics|other",
     "text":"short phrase (<=6 words)",
     "priority":"high|medium|low" }}
]}}

Important: Return only valid JSON, no additional text or explanation.
"""

    # Create message and get response
    message = HumanMessage(content=prompt)
    response = llm.invoke([message])
    raw = response.content

    # Extract JSON from response
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        raise ValueError(f"No JSON found in Groq output. Raw response: {raw}")
    
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in response: {e}. Raw: {raw}")

def test_available_models():
    """Test which Gemma models are available through your Groq API key"""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("❌ GROQ_API_KEY not set")
        return
    
    gemma_models = [
        "gemma2-9b-it",
        "gemma-7b-it"
    ]
    
    print("Testing available Gemma models:")
    for model in gemma_models:
        try:
            llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name=model,
                temperature=0
            )
            # Test with a simple query
            response = llm.invoke([HumanMessage(content="Hello")])
            print(f"✓ {model} - Available")
        except Exception as e:
            print(f"✗ {model} - Error: {str(e)}")

if __name__ == "__main__":
    import sys
    
    # Check if running in Jupyter/Colab (has -f argument)
    if any('-f' in arg for arg in sys.argv):
        print("🔧 Running in notebook mode")
        print("\n" + "="*50)
        print("TESTING AVAILABLE MODELS")
        print("="*50)
        test_available_models()
        
        print("\n" + "="*50) 
        print("EXAMPLE USAGE")
        print("="*50)
        example_query = "what are the latest gst reforms in india in 2025"
        print(f"Query: {example_query}")
        try:
            out = planner(example_query, depth="deep")
            print(json.dumps(out, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("query", nargs="+", help="High-level query")
        parser.add_argument("--model", default="gemma2-9b-it", 
                           choices=["gemma2-9b-it", "gemma-7b-it"],
                           help="Gemma model to use")
        parser.add_argument("--depth", default="normal", 
                           choices=["shallow", "normal", "deep"],
                           help="Depth of decomposition")
        parser.add_argument("--test-models", action="store_true",
                           help="Test available Gemma models")
        
        args = parser.parse_args()
        
        if args.test_models:
            test_available_models()
        else:
            q = " ".join(args.query)
            try:
                out = planner(q, model=args.model, depth=args.depth)
                print(json.dumps(out, indent=2, ensure_ascii=False))
            except Exception as e:
                print(f"❌ Error: {e}")
                print("\nTry running with --test-models to check available models")
