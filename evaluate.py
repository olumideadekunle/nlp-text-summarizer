"""
Evaluates both extractive and abstractive summarization using ROUGE scores
on a set of built-in sample texts with reference summaries.

Usage: python evaluate.py
"""

import pandas as pd
from rouge_score import rouge_scorer
from summarizer import extractive_summary, abstractive_summary

SAMPLES = [
    {
        "text": (
            "The Apollo 11 mission was the first crewed lunar landing. "
            "On July 20, 1969, Neil Armstrong and Buzz Aldrin landed on the Moon "
            "while Michael Collins orbited above. Armstrong became the first human "
            "to walk on the lunar surface, declaring 'That's one small step for man, "
            "one giant leap for mankind.' The mission returned safely on July 24, 1969."
        ),
        "reference": "Neil Armstrong and Buzz Aldrin landed on the Moon on July 20, 1969, with Armstrong becoming the first human to walk on its surface."
    },
    {
        "text": (
            "Artificial intelligence is transforming industries worldwide. "
            "From healthcare to finance, AI automates tasks and uncovers insights from data. "
            "Machine learning enables systems to learn from data without explicit programming. "
            "Deep learning uses neural networks to solve complex problems like image recognition "
            "and natural language processing."
        ),
        "reference": "AI is transforming industries by automating tasks and enabling machine learning and deep learning to solve complex problems."
    },
    {
        "text": (
            "Climate change is one of the most pressing global issues. "
            "Rising temperatures are melting glaciers and causing sea levels to rise. "
            "Extreme weather events are becoming more frequent and severe. "
            "Scientists urge significant reductions in greenhouse gas emissions. "
            "Governments worldwide are transitioning to renewable energy sources."
        ),
        "reference": "Climate change is causing rising temperatures, melting glaciers, and extreme weather, prompting global efforts to reduce emissions and adopt renewable energy."
    },
]

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

rows = []
print("Evaluating summaries with ROUGE scores...\n")

for i, sample in enumerate(SAMPLES):
    text      = sample["text"]
    reference = sample["reference"]

    ext  = extractive_summary(text, num_sentences=2)
    print(f"Sample {i+1} — running abstractive (BART)...")
    abst = abstractive_summary(text, max_length=80, min_length=20)

    ext_scores  = scorer.score(reference, ext)
    abst_scores = scorer.score(reference, abst)

    rows.append({
        "SampleID":             i + 1,
        "Method":               "Extractive",
        "ROUGE-1":              round(ext_scores["rouge1"].fmeasure, 4),
        "ROUGE-2":              round(ext_scores["rouge2"].fmeasure, 4),
        "ROUGE-L":              round(ext_scores["rougeL"].fmeasure, 4),
        "Summary":              ext,
    })
    rows.append({
        "SampleID":             i + 1,
        "Method":               "Abstractive",
        "ROUGE-1":              round(abst_scores["rouge1"].fmeasure, 4),
        "ROUGE-2":              round(abst_scores["rouge2"].fmeasure, 4),
        "ROUGE-L":              round(abst_scores["rougeL"].fmeasure, 4),
        "Summary":              abst,
    })

df = pd.DataFrame(rows)

print("\n── ROUGE Evaluation Results ──")
print(df[["SampleID", "Method", "ROUGE-1", "ROUGE-2", "ROUGE-L"]].to_string(index=False))

print("\n── Average Scores by Method ──")
print(df.groupby("Method")[["ROUGE-1", "ROUGE-2", "ROUGE-L"]].mean().round(4))

df.to_csv("evaluation_results.csv", index=False)
print("\nFull results saved to evaluation_results.csv")
