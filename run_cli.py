"""
Command-line interface for the NLP Text Summarizer.
Run without needing Streamlit — useful for scripting or batch text processing.

Usage:
    python run_cli.py --text "Your long text here..." --mode extractive
    python run_cli.py --text "Your long text here..." --mode abstractive
    python run_cli.py --text "Your long text here..." --mode both
    python run_cli.py --file input.txt --mode both
"""

import argparse
import os
from summarizer import extractive_summary, abstractive_summary, export_to_csv

def main():
    parser = argparse.ArgumentParser(description="NLP Text Summarizer CLI")
    parser.add_argument("--text", type=str, help="Text to summarize")
    parser.add_argument("--file", type=str, help="Path to a .txt file to summarize")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["extractive", "abstractive", "both"],
                        help="Summarization mode (default: both)")
    parser.add_argument("--sentences", type=int, default=3,
                        help="Number of sentences for extractive summary (default: 3)")
    parser.add_argument("--max_length", type=int, default=130,
                        help="Max length for abstractive summary (default: 130)")
    parser.add_argument("--min_length", type=int, default=30,
                        help="Min length for abstractive summary (default: 30)")
    parser.add_argument("--output", type=str, default="output.csv",
                        help="Output CSV file path (default: output.csv)")
    args = parser.parse_args()

    # ── Get input text ────────────────────────────────────────────────────────
    if args.file:
        if not os.path.exists(args.file):
            raise FileNotFoundError(f"File not found: {args.file}")
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read().strip()
        print(f"Loaded text from {args.file} ({len(text)} characters)\n")
    elif args.text:
        text = args.text.strip()
    else:
        # Built-in sample text for quick demo
        text = (
            "Artificial intelligence is transforming industries worldwide. "
            "From healthcare to finance, AI is being used to automate tasks, improve accuracy, "
            "and uncover insights from massive datasets. Machine learning, a subset of AI, "
            "enables systems to learn patterns from data without being explicitly programmed. "
            "Deep learning uses neural networks with many layers to solve complex problems "
            "such as image recognition, speech processing, and natural language understanding. "
            "As AI continues to advance, its impact on society, jobs, and ethics becomes "
            "an increasingly important topic of discussion among researchers and policymakers."
        )
        print("No input provided — using built-in sample text.\n")

    print(f"Original Text ({len(text.split())} words):")
    print(f"{text}\n")
    print("=" * 60)

    record = {"OriginalText": text}

    # ── Extractive ────────────────────────────────────────────────────────────
    if args.mode in ("extractive", "both"):
        ext = extractive_summary(text, args.sentences)
        print(f"\nExtractive Summary ({args.sentences} sentences):")
        print(ext)
        record["ExtractiveSummary"] = ext

    # ── Abstractive ───────────────────────────────────────────────────────────
    if args.mode in ("abstractive", "both"):
        print("\nRunning BART model (first run downloads ~1.6 GB)...")
        abst = abstractive_summary(text, args.max_length, args.min_length)
        print(f"\nAbstractive Summary:")
        print(abst)
        record["AbstractiveSummary"] = abst

    print("\n" + "=" * 60)

    # ── Export ────────────────────────────────────────────────────────────────
    export_to_csv([record], path=args.output)

if __name__ == "__main__":
    main()
