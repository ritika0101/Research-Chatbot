# Research Chatbot

## What It Does

- **Automated Research Assistant** – Converts complex user queries into structured, source-backed research reports.  
- **Evidence-Backed Insights** – Every claim is supported with citations and reliable source links.  
- **Multi-Source Synthesis** – Gathers and merges data from multiple credible web sources.  
- **Noise Reduction** – Filters out low-quality, repetitive, or irrelevant content automatically.  
- **User-Friendly Output** – Delivers clean Markdown and JSON reports for seamless integration.  
- **Configurable Depth** – Choose between concise summaries or deep, comprehensive research.  
- **Contradiction Alerts** – Flags conflicting information between sources for transparency.  
- **Interactive Chat** – Lets users ask follow-up questions after the report using a contextual RAG-based chatbot.

---

## Architecture
<img width="400" height="600" alt="architecture" src="https://github.com/user-attachments/assets/212051b8-b42c-4d03-bb74-ac6c0910f5d5" />

---

**Key Features:**
- Iterative refinement loop for irrelevant data.  
- Combines multi-source data for cohesive analysis.  
- Embedding-driven retrieval for post-report Q&A.  

---

## Tech Stack

**Core and Data Collection**
- Python  
- Google API (Gemini Flash 2.5)  
- SerpAPI  
- BeautifulSoup  
- Firecrawl  
- Groq AI  
- LangChain / LangChain Community  

**Document Processing**
- FAISS  
- Sentence Transformers  

**Output & UI**
- Streamlit (Frontend)  
- Markdown / JSON Reports  

---
