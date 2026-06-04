# NLP Text Summarizer

A web app that summarizes text using two approaches:
- **Extractive** — picks the most important sentences using TF-IDF scoring (NLTK + scikit-learn)
- **Abstractive** — generates a new summary using Facebook's BART model (Hugging Face Transformers)

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

> First time you use the Abstractive tab, BART (~1.6 GB) will download and cache automatically.

## Tech Stack

- [Streamlit](https://streamlit.io/) — UI
- [NLTK](https://www.nltk.org/) — sentence tokenization
- [scikit-learn](https://scikit-learn.org/) — TF-IDF vectorization
- [Hugging Face Transformers](https://huggingface.co/docs/transformers) — BART model (`facebook/bart-large-cnn`)
