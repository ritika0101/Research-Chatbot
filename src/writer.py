"""
Step 4: Writer — Markdown report (Groq API)
"""

import os, json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

def writer(findings, contradictions, model="meta-llama/llama-4-scout-17b-16e-instruct"):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("❌ GROQ_API_KEY not set in .env")

    client = Groq(api_key=api_key)
    
    prompt = f"""
You are WriterAgent. Use findings and contradictions to create a concise Markdown report.
Rules:
- Every fact must be followed by [url].
- Use professional, clear style.
- Mention contradictions if any.

Findings:
{json.dumps(findings, ensure_ascii=False)}

Contradictions:
{json.dumps(contradictions, ensure_ascii=False)}

Markdown report:
"""

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
            max_tokens=1024,
            top_p=1,
            stream=False,
            stop=None
        )
        
        return completion.choices[0].message.content
        
    except Exception as e:
        raise RuntimeError(f"❌ Groq API error: {str(e)}")


def writer_streaming(findings, contradictions, model="meta-llama/llama-3.1-70b-versatile"):
    """Alternative streaming version"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("❌ GROQ_API_KEY not set in .env")

    client = Groq(api_key=api_key)
    
    prompt = f"""
You are WriterAgent. Use findings and contradictions to create a concise Markdown report.
Rules:
- Every fact must be followed by [url].
- Use professional, clear style.
- Mention contradictions if any.

Findings:
{json.dumps(findings, ensure_ascii=False)}

Contradictions:
{json.dumps(contradictions, ensure_ascii=False)}

Markdown report:
"""

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
            max_tokens=1024,
            top_p=1,
            stream=True,
            stop=None
        )
        
        result = ""
        for chunk in completion:
            content = chunk.choices[0].delta.content or ""
            result += content
            print(content, end="")
        
        print()  # New line after streaming
        return result
        
    except Exception as e:
        raise RuntimeError(f"❌ Groq API error: {str(e)}")


if __name__ == "__main__":
    demo_findings = [
        {"subq_id":"Q1","claim":"H1B fee raised to $100,000","url":"https://example.com","nums":["100000"]}
    ]
    
    print("=== Non-streaming version ===")
    print(writer(demo_findings, []))
    
    print("\n=== Streaming version ===")
    writer_streaming(demo_findings, [])