import streamlit as st
from summarizer import extractive_summary, abstractive_summary, export_to_csv

st.set_page_config(page_title="NLP Text Summarizer", layout="centered")
st.title("NLP Text Summarizer")
st.caption("Extractive (TF-IDF) · Abstractive (BART)")

text = st.text_area("Paste your text here", height=250, placeholder="Enter at least a few sentences...")

tab1, tab2 = st.tabs(["Extractive", "Abstractive"])

with tab1:
    num_sentences = st.slider("Number of sentences", 1, 10, 3)
    if st.button("Summarize (Extractive)"):
        if text.strip():
            with st.spinner("Scoring sentences with TF-IDF..."):
                result = extractive_summary(text, num_sentences)
            st.subheader("Extractive Summary")
            st.write(result)
            export_to_csv([{"OriginalText": text, "ExtractiveSummary": result}])
        else:
            st.warning("Please enter some text.")

with tab2:
    col1, col2 = st.columns(2)
    max_len = col1.number_input("Max length", 50, 500, 130)
    min_len = col2.number_input("Min length", 10, 100, 30)
    if st.button("Summarize (Abstractive)"):
        if text.strip():
            with st.spinner("Running BART model (first run downloads ~1.6 GB)..."):
                result = abstractive_summary(text, int(max_len), int(min_len))
            st.subheader("Abstractive Summary")
            st.write(result)
            export_to_csv([{"OriginalText": text, "AbstractiveSummary": result}])
        else:
            st.warning("Please enter some text.")
