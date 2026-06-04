import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from transformers import BartForConditionalGeneration, BartTokenizer

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

def extractive_summary(text: str, num_sentences: int = 3) -> str:
    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return text

    vectorizer = TfidfVectorizer(stop_words=stopwords.words("english"))
    tfidf_matrix = vectorizer.fit_transform(sentences)
    scores = np.array(tfidf_matrix.sum(axis=1)).flatten()

    top_indices = sorted(np.argsort(scores)[-num_sentences:])
    return " ".join(sentences[i] for i in top_indices)

_model = None
_tokenizer = None

def _load_model():
    global _model, _tokenizer
    if _model is None:
        _tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        _model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

def abstractive_summary(text: str, max_length: int = 130, min_length: int = 30) -> str:
    _load_model()
    inputs = _tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    ids = _model.generate(inputs["input_ids"], max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4)
    return _tokenizer.decode(ids[0], skip_special_tokens=True)
