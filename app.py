"""
Streamlit app entrypoint for SwiftVisa AI - Based Visa eligibility Screening Agent
Lightweight RAG MVP integrating Chroma + MiniLM embeddings + LLM (local or API)
"""

import streamlit as st
import os
from datetime import datetime
from rag.rag_engine import run_rag
from rag.retriever import load_retriever
from models.local_llm import load_local_llm
from models.api_llm import openai_callable
from utils.report_generator import compose_report_text
from utils.pdf_utils import text_to_pdf

st.set_page_config(page_title="SwiftVisa AI - Visa Eligibility", layout="wide")
st.title("SwiftVisa AI — Visa Eligibility Screening (MVP)")

st.markdown("**Countries:** UK, USA, Canada • **Visa types:** Student, Skilled Worker, Visit/Tourist, Dependent")

with st.sidebar.expander("LLM Settings"):
    use_api = st.checkbox("Use OpenAI API (if unchecked, app will try local LLM)", value=False)
    api_model = st.text_input("OpenAI model (when using API)", value="gpt-3.5-turbo")
    api_key = st.text_input("OpenAI API key (or set OPENAI_API_KEY env var)", type="password")

with st.form("profile"):
    country = st.selectbox("Country", ["UK", "USA", "Canada"])
    visa_type = st.selectbox("Visa Type", ["Student", "Skilled Worker", "Visit/Tourist", "Dependent"])
    age = st.number_input("Age", min_value=0, max_value=120, value=25)
    nationality = st.text_input("Nationality", value="India")
    ielts = st.text_input("IELTS (or equivalent) score (enter N/A if none)", value="6.5")
    funds = st.text_input("Proof of funds (amount & currency)", value="£8000")
    extra = st.text_area("Extra info (e.g., CAS / Offer letter / Work exp)", value="CAS: yes")
    submitted = st.form_submit_button("Check Eligibility")

if submitted:
    user_profile = {
        "country": country,
        "visa_type": visa_type,
        "age": age,
        "nationality": nationality,
        "ielts": ielts,
        "funds": funds,
        "extra": extra
    }

    st.info("Running retrieval + reasoning...")
    # Choose LLM callable
    llm_callable = None
    if use_api:
        try:
            if api_key.strip():
                llm_callable = openai_callable(api_key, model=api_model)
            else:
                llm_callable = openai_callable(model=api_model)
        except Exception as e:
            st.error(f"OpenAI API LLM error: {e}")
            llm_callable = None
    else:
        try:
            # Default local model name - replace if you have a different local model
            llm_callable = load_local_llm(model_name="togethercomputer/phi-2")
        except Exception as e:
            st.warning(f"Local LLM load failed: {e}")
            st.info("Falling back to OpenAI API LLM. Toggle 'Use OpenAI API' in sidebar to provide API key.")
            llm_callable = None

    if llm_callable is None:
        st.stop()

    try:
        result_text, docs = run_rag(user_profile, llm_callable, k=6)
    except Exception as e:
        st.error(f"RAG execution error: {e}")
        raise

    st.subheader("Eligibility Result (Formatted)")
    st.markdown("**Result:**")
    st.code(result_text, language="text")

    st.subheader("Top Retrieved Policy Snippets")
    for i, d in enumerate(docs[:6]):
        md = d.metadata or {}
        st.markdown(f"**[{i+1}] Source:** {md.get('source','unknown')} — *{md.get('country','')} / {md.get('visa_type','')}*")
        preview = d.page_content[:800].replace("\n", " ")
        st.write(preview + ("..." if len(d.page_content) > 800 else ""))

    # Compose report and enable download
    report_text = compose_report_text(result_text, user_profile, docs)
    if st.button("Generate and Download PDF"):
        fname = f"visa_report_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.pdf"
        out_path = os.path.join("vector_db", fname)
        text_to_pdf(report_text, out_path, title="SwiftVisa AI — Visa Eligibility Report")
        with open(out_path, "rb") as f:
            st.download_button("Download PDF", f, file_name=fname, mime="application/pdf")

    st.success("Done. Review the result and sources above.")
