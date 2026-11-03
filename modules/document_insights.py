"""
Document statistics and charts.

Counting words/sentences is simple but useful: it tells you how big the document is and
(together with OCR confidence) gives a quick sanity check on extraction quality.
Charts use Matplotlib + Seaborn with a dark-navy theme that matches the Streamlit app.
"""
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")                      # no GUI window needed; Streamlit displays the figure
import matplotlib.pyplot as plt
import seaborn as sns

from modules.text_preprocessor import (remove_stop_words, split_paragraphs, split_sentences,
                                       tokenize)

# Theme colours shared with the app
NAVY, PANEL = "#0A1128", "#111C3A"
BLUE, CYAN = "#3B82F6", "#22D3EE"
TEXT, MUTED, GRID = "#E2E8F0", "#94A3B8", "#1E2B4F"

WORDS_PER_MINUTE = 200                      # average silent reading speed


def top_keywords(text: str, n: int = 15) -> List[Tuple[str, int]]:
    """Most frequent meaningful words (stop-words removed). Frequency-based, not semantic."""
    counts = Counter(remove_stop_words(tokenize(text)))
    return counts.most_common(n)


def compute_insights(text: str) -> Dict:
    """All headline numbers shown on the Insights page."""
    words = re.findall(r"\b\w[\w'-]*\b", text)
    sentences = split_sentences(text)
    paragraphs = split_paragraphs(text)
    sentence_lengths = [len(s.split()) for s in sentences]
    lowered = [w.lower() for w in words]
    unique = len(set(lowered))
    return {
        "words": len(words),
        "characters": len(text),
        "characters_no_spaces": len(re.sub(r"\s", "", text)),
        "paragraphs": len(paragraphs),
        "sentences": len(sentences),
        "unique_words": unique,
        "lexical_diversity": (unique / len(words)) if words else 0.0,   # 1.0 = every word is different
        "avg_sentence_length": (sum(sentence_lengths) / len(sentence_lengths)) if sentence_lengths else 0.0,
        "avg_word_length": (sum(len(w) for w in words) / len(words)) if words else 0.0,
        "reading_time_min": len(words) / WORDS_PER_MINUTE,
        "keywords": top_keywords(text, 15),
        "sentence_lengths": sentence_lengths,
    }


def _style_axes(fig, ax):
    """Apply the dark theme to a figure/axes."""
    fig.patch.set_facecolor(PANEL)
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def keyword_chart(keywords: List[Tuple[str, int]]) -> Optional[plt.Figure]:
    """Horizontal bar chart of the most frequent keywords."""
    if not keywords:
        return None
    words, counts = zip(*keywords)
    fig, ax = plt.subplots(figsize=(6.5, 0.38 * len(words) + 1.2))
    sns.barplot(x=list(counts), y=list(words), color=CYAN, ax=ax)
    ax.set_title("Most frequent keywords", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("Occurrences")
    ax.set_ylabel("")
    _style_axes(fig, ax)
    fig.tight_layout()
    return fig


def sentence_length_chart(lengths: List[int]) -> Optional[plt.Figure]:
    """Histogram of sentence lengths (in words)."""
    if not lengths:
        return None
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    sns.histplot(lengths, bins=min(20, max(5, len(set(lengths)))), color=BLUE, edgecolor=PANEL, ax=ax)
    ax.set_title("Sentence length distribution", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("Words per sentence")
    ax.set_ylabel("Sentences")
    _style_axes(fig, ax)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    fig.tight_layout()
    return fig


def probability_chart(probabilities: Dict[str, float]) -> Optional[plt.Figure]:
    """Bar chart of the classifier's probability for each category (top bar highlighted)."""
    if not probabilities:
        return None
    labels = list(probabilities.keys())
    values = [probabilities[k] * 100 for k in labels]
    colors = [CYAN if i == 0 else BLUE for i in range(len(labels))]
    fig, ax = plt.subplots(figsize=(6.5, 0.45 * len(labels) + 1.1))
    sns.barplot(x=values, y=labels, hue=labels, palette=dict(zip(labels, colors)), legend=False, ax=ax)
    for i, v in enumerate(values):
        ax.text(v + 1, i, f"{v:.1f}%", va="center", color=TEXT, fontsize=9)
    ax.set_xlim(0, 110)
    ax.set_title("Category confidence", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("Probability (%)")
    ax.set_ylabel("")
    _style_axes(fig, ax)
    fig.tight_layout()
    return fig


def term_weight_chart(top_terms: List[Dict]) -> Optional[plt.Figure]:
    """Bar chart of the words that pushed the classifier toward its answer."""
    if not top_terms:
        return None
    terms = [t["term"] for t in top_terms]
    weights = [t["weight"] for t in top_terms]
    fig, ax = plt.subplots(figsize=(6.5, 0.4 * len(terms) + 1.1))
    sns.barplot(x=weights, y=terms, color=BLUE, ax=ax)
    ax.set_title("Words that influenced the prediction", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("Influence (model weight x TF-IDF)")
    ax.set_ylabel("")
    _style_axes(fig, ax)
    fig.tight_layout()
    return fig


def confusion_matrix_chart(matrix: List[List[int]], labels: List[str]) -> plt.Figure:
    """Heatmap of the held-out confusion matrix (rows = true label, columns = predicted)."""
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap=sns.light_palette(CYAN, as_cmap=True),
                xticklabels=labels, yticklabels=labels, cbar=False, linewidths=0.5,
                linecolor=PANEL, ax=ax, annot_kws={"color": NAVY, "fontsize": 10})
    ax.set_title("Confusion matrix (held-out test split)", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    _style_axes(fig, ax)
    ax.grid(False)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    return fig
