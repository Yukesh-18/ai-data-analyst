"""
STEP 5 (POLISHED): THE AI DATA ANALYST APP
=========================================
This is the improved version of your app — same core logic as before,
just with a nicer layout, sample questions, and quick stats.

Run with:
    python -m streamlit run app.py
"""

import streamlit as st
import pandas as pd
import ollama

# ---------------------------------------------------------------
# PAGE CONFIG - sets the browser tab title, icon, and page width
# ---------------------------------------------------------------
st.set_page_config(
    page_title="AI Data Analyst Agent",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------------------------------------
# HEADER SECTION
# ---------------------------------------------------------------
st.title("📊 AI Data Analyst Agent")
st.markdown(
    "Upload any CSV file and ask questions about it in plain English. "
    "This app combines a classic data-analysis pipeline with a local "
    "AI model (Llama 3.2 via Ollama) to explain your data — no data "
    "ever leaves your computer."
)

st.divider()

# ---------------------------------------------------------------
# SIDEBAR - project info, so it looks like a real deployed tool
# ---------------------------------------------------------------
with st.sidebar:
    st.header("About this project")
    st.markdown(
        """
        **What it does:**
        - Upload a CSV dataset
        - Get instant summary statistics
        - Ask natural-language questions about your data
        - Get AI-generated answers, powered by a locally-run LLM

        **Tech stack:**
        - `pandas` for data handling
        - `Ollama` + `Llama 3.2` for the AI layer
        - `Streamlit` for the web interface

        **Privacy note:** since the AI model runs locally on this
        machine, your uploaded data is never sent to an external server.
        """
    )
    st.divider()
    st.caption("Built as a Data Science + Gen AI portfolio project.")

# ---------------------------------------------------------------
# FILE UPLOAD SECTION
# ---------------------------------------------------------------
uploaded_file = st.file_uploader("📁 Upload a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # ---- Quick stats row, shown as metric cards ----
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", f"{df.shape[1]:,}")
    col3.metric("Missing values", f"{df.isnull().sum().sum():,}")

    st.divider()

    # ---- Data preview and summary, side by side ----
    tab1, tab2 = st.tabs(["🔍 Preview", "📈 Summary Statistics"])
    with tab1:
        st.dataframe(df.head(10), use_container_width=True)
    with tab2:
        st.dataframe(df.describe(), use_container_width=True)

    st.divider()

    # ---------------------------------------------------------------
    # QUESTION SECTION
    # ---------------------------------------------------------------
    st.subheader("💬 Ask a question about your data")

    # Sample questions the user can click instead of typing from scratch.
    # We use st.session_state so clicking a button fills the text box.
    if "question_text" not in st.session_state:
        st.session_state.question_text = ""

    st.write("Try one of these, or type your own:")
    sample_cols = st.columns(3)
    sample_questions = [
        "What's the average value in each numeric column?",
        "Which column has the most missing values?",
        "Summarize the overall shape of this dataset.",
    ]
    for i, sample_q in enumerate(sample_questions):
        if sample_cols[i].button(sample_q, use_container_width=True):
            st.session_state.question_text = sample_q

    question = st.text_input(
        "Your question",
        value=st.session_state.question_text,
        placeholder="e.g. What trends do you see in this data?"
    )

    if question:
        summary = df.describe().to_string()
        prompt = f"""
        Here is a summary of a dataset:
        {summary}

        Based on this data, answer the following question as clearly
        and concisely as possible:
        {question}
        """

        with st.spinner("Analyzing your data..."):
            response = ollama.chat(model='llama3.2', messages=[
                {'role': 'user', 'content': prompt}
            ])
            answer = response['message']['content']

        st.success("Analysis complete")
        st.markdown(f"**Answer:** {answer}")

else:
    # Shown before any file is uploaded, so the page doesn't look empty
    st.info("👆 Upload a CSV file above to get started.")

# ---------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------
st.divider()
st.caption("AI Data Analyst Agent — built with pandas, scikit-learn, Ollama, and Streamlit.")