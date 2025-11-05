import os
import json
import streamlit as st
from datetime import datetime
from pathlib import Path

# Import your pipeline modules (assumes they are in same folder)
from planner import planner
from scout import scout
from analyst import analyst
from writer import writer

# Import the RAGChatbot class from chatbot.py
from chatbot import RAGChatbot  # <-- Ensure chatbot.py is accessible and contains RAGChatbot class

st.set_page_config(page_title="Project Galileo — Research Agent", layout="wide")

st.title("Project Galileo — Research Agent")
st.markdown("Enter a research question, run the pipeline, and get an evidence-backed report.")

# Sidebar controls
st.sidebar.header("Settings")
query = st.sidebar.text_area("Research query", value="What is the current status of H1B visa in 2025?", height=80)
depth = st.sidebar.selectbox("Depth", ["normal", "shallow", "deep"], index=1)
top_k = st.sidebar.slider("Top URLs per keyword (Scout)", min_value=1, max_value=4, value=2)
keywords_per_subq = st.sidebar.slider("Keywords per sub-question", min_value=1, max_value=3, value=2)
save_report_file = st.sidebar.checkbox("Save report to file", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("⚠️ Make sure `.env` contains `GEMINI_API_KEY` and `GROQ_API_KEY` before running.")

# Main area columns for controls and quick actions
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Run pipeline")
    run_btn = st.button("Run Research Agent")

with col_right:
    st.subheader("Quick actions")
    if st.button("Show latest planner output (planner_output.json)"):
        try:
            with open("planner_output.json", "r", encoding="utf-8") as f:
                st.code(json.dumps(json.load(f), indent=2, ensure_ascii=False), language="json")
        except Exception:
            st.info("No planner_output.json found yet. Run the pipeline first.")

# Placeholders for progress / outputs
pln_ph = st.empty()
scout_ph = st.empty()
analyst_ph = st.empty()
writer_ph = st.empty()

report_filepath = None  # will hold path if report is saved

# Initialize session state for chatbot and documents
if 'chatbot_ready' not in st.session_state:
    st.session_state['chatbot_ready'] = False

if 'latest_report' not in st.session_state:
    st.session_state['latest_report'] = None

# ---------------- Pipeline Execution ---------------- #
if run_btn:
    if not query.strip():
        st.error("Please enter a research query.")
    else:
        # Step 1: Planner
        pln_ph.info("Step 1 — Planner: contacting Gemini...")
        try:
            plan = planner(query, depth=depth)
        except Exception as e:
            st.error(f"Planner error: {e}")
            raise
        subqs = plan.get("subquestions", [])
        pln_ph.success(f"Planner produced {len(subqs)} sub-questions.")

        st.subheader("Planner — Sub-questions")
        st.json(plan)

        # save planner output
        with open("planner_output.json", "w", encoding="utf-8") as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)

        # Step 2: Scout
        scout_ph.info("Step 2 — Scout: searching & scraping (DuckDuckGo)...")
        try:
            results = scout(subqs, top_k=top_k)
        except TypeError:
            results = scout(subqs)  # fallback
        scout_ph.success(f"Scout found {len(results)} items.")

        st.subheader("Scout — top scraped results (first 8)")
        if results:
            for r in results[:8]:
                st.markdown(
                    f"**{r.get('title','(no title)')}**  \n{r.get('url')}  \n_{r.get('subq_id')}: {r.get('search_keyword', r.get('keyword',''))}_"
                )
                snippet = (r.get("text") or "").strip().replace("\n", " ")[:600]
                st.write(snippet + ("…" if len(snippet) >= 600 else ""))
                st.markdown("---")
        else:
            st.info("No scraped results found. Try increasing Top URLs or keywords per sub-question.")

        # Step 3: Analyst
        analyst_ph.info("Step 3 — Analyst: extracting claims & checking contradictions...")
        try:
            findings, contradictions = analyst(results)
        except Exception as e:
            st.error(f"Analyst error: {e}")
            raise
        analyst_ph.success(f"Analyst found {len(findings)} findings and {len(contradictions)} contradictions.")

        st.subheader("Analyst — Top findings (first 10)")
        if findings:
            for f in findings[:10]:
                claim = f.get("claim", "").replace("\n", " ")
                st.markdown(f"- **{f.get('subq_id')}**: {claim}  \n  — Source: {f.get('url')}")
        else:
            st.info("No findings.")

        if contradictions:
            st.subheader("Detected contradictions")
            st.json(contradictions)

        # Step 4: Writer
        writer_ph.info("Step 4 — Writer: generating Markdown report (LangChain + Gemini)...")
        try:
            md_report = writer(findings, contradictions)
        except Exception as e:
            writer_ph.error(f"Writer error: {e}")
            raise
        writer_ph.success("Report generated.")

        st.subheader("Final Markdown Report")
        st.markdown("---")
        st.markdown(md_report)

        # Save report if requested
        if save_report_file:
            fn = f"report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.md"
            with open(fn, "w", encoding="utf-8") as f:
                f.write(md_report)
            st.success(f"Saved report to {fn}")
            report_filepath = fn
            st.session_state['latest_report'] = fn
            
            with open(fn, "rb") as f:
                st.download_button("Download report (.md)", f, file_name=fn)

        # Initialize and index chatbot with the new report
        st.info("Initializing chatbot with the generated report...")
        try:
            groq_api_key = os.getenv("GROQ_API_KEY", "your-groq-api-key-here")
            
            if groq_api_key == "your-groq-api-key-here":
                st.error("⚠️ GROQ_API_KEY not found in environment variables. Please set it in your .env file.")
                st.session_state['chatbot_ready'] = False
            else:
                # Create new chatbot instance
                chatbot = RAGChatbot(groq_api_key=groq_api_key)
                
                # Index the report file if it exists
                if report_filepath and Path(report_filepath).exists():
                    chatbot.load_and_index_documents([report_filepath])
                    st.session_state['chatbot'] = chatbot
                    st.session_state['chatbot_ready'] = True
                    st.success("✅ Chatbot successfully indexed the report and is ready for questions!")
                else:
                    st.warning("Report file not found. Chatbot may not have proper context.")
                    st.session_state['chatbot_ready'] = False
                    
        except Exception as e:
            st.error(f"Error initializing chatbot: {e}")
            st.session_state['chatbot_ready'] = False

# ---------------- Chatbot Section ---------------- #
st.markdown("---")
st.subheader("Step 5 — Ask the Research Agent Chatbot")

# Check if chatbot is ready
if not st.session_state.get('chatbot_ready', False):
    st.warning("⚠️ Chatbot not ready yet. Run the pipeline first to generate a report that the chatbot can use for context.")
    
    # Offer to initialize with existing report if available
    if st.session_state.get('latest_report') and Path(st.session_state['latest_report']).exists():
        if st.button("Initialize Chatbot with Latest Report"):
            try:
                groq_api_key = os.getenv("GROQ_API_KEY", "your-groq-api-key-here")
                if groq_api_key != "your-groq-api-key-here":
                    chatbot = RAGChatbot(groq_api_key=groq_api_key)
                    chatbot.load_and_index_documents([st.session_state['latest_report']])
                    st.session_state['chatbot'] = chatbot
                    st.session_state['chatbot_ready'] = True
                    st.success("✅ Chatbot initialized with existing report!")
                    st.experimental_rerun()
                else:
                    st.error("GROQ_API_KEY not set in environment variables.")
            except Exception as e:
                st.error(f"Error initializing chatbot: {e}")
else:
    # Chatbot is ready - show interface
    chatbot = st.session_state['chatbot']
    
    # User question input
    user_question = st.text_input("Ask a question about the research:")
    
    if user_question:
        with st.spinner("Chatbot is generating an answer..."):
            try:
                response = chatbot.ask(user_question)
                
                st.markdown(f"**Answer:** {response['answer']}")
                
                if response['sources']:
                    st.markdown(f"**Sources ({len(response['sources'])}):**")
                    for i, src in enumerate(response['sources'], 1):
                        snippet = src['content']
                        st.markdown(f"{i}. `{src['metadata'].get('file_name','unknown')}`: {snippet}")
                else:
                    st.info("No sources found for this answer.")
                    
            except Exception as e:
                st.error(f"Error getting chatbot response: {e}")

# Show current status
st.sidebar.markdown("---")
st.sidebar.markdown("**Chatbot Status:**")
if st.session_state.get('chatbot_ready', False):
    st.sidebar.success("✅ Ready")
    if st.session_state.get('latest_report'):
        st.sidebar.info(f"Using: {Path(st.session_state['latest_report']).name}")
else:
    st.sidebar.error("❌ Not Ready")

st.markdown("---")
st.caption("Project Galileo — Streamlit frontend. Uses Gemini for LLM steps, Groq for chatbot, and DuckDuckGo for search.")