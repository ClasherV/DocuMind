"""
Extractive text summarisation (TextRank over TF-IDF sentence vectors).

"Extractive" means the summary is made of the document's OWN sentences, picked because
they are the most "central". Nothing is rewritten or invented - which is safe, but it also
means the result can feel choppy and may miss ideas that are spread across many sentences.
This is NOT comparable to a generative AI model that paraphrases.

How TextRank works (beginner version):
  1. Turn every sentence into a TF-IDF vector.
  2. Measure how similar every pair of sentences is (cosine similarity).
  3. Treat sentences as nodes in a graph; similarity = strength of the link between them.
  4. Run PageRank: a sentence is important if many important sentences are similar to it.
  5. Keep the top-scoring sentences, then put them back in original reading order.

Long documents: the similarity matrix grows with the square of the sentence count, so the
document is cut into windows of WINDOW sentences, each window is ranked separately, and each
window contributes sentences in proportion to its size. This also spreads the summary across
the whole document instead of clustering it in one section.
"""
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from modules.text_preprocessor import preprocess_for_model, split_sentences

WINDOW = 150            # sentences ranked together
MIN_WORDS = 5           # shorter "sentences" (headings, page numbers) are never selected
DAMPING = 0.85          # standard PageRank damping factor


def _pagerank(similarity: np.ndarray, iterations: int = 60, tolerance: float = 1e-6) -> np.ndarray:
    """Plain NumPy PageRank (power iteration) on a similarity matrix."""
    n = similarity.shape[0]
    np.fill_diagonal(similarity, 0.0)                       # a sentence does not vote for itself
    row_sums = similarity.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0                           # avoid division by zero
    transition = similarity / row_sums
    scores = np.full(n, 1.0 / n)
    for _ in range(iterations):
        updated = (1 - DAMPING) / n + DAMPING * transition.T @ scores
        if np.abs(updated - scores).sum() < tolerance:
            scores = updated
            break
        scores = updated
    return scores


def _rank_window(sentences: List[str]) -> np.ndarray:
    """Importance score for every sentence in one window (higher = more central)."""
    processed = [preprocess_for_model(s) for s in sentences]
    if sum(1 for p in processed if p) < 2:
        return np.ones(len(sentences))
    try:
        vectors = TfidfVectorizer(sublinear_tf=True).fit_transform(processed)
    except ValueError:                                       # empty vocabulary
        return np.ones(len(sentences))
    scores = _pagerank(cosine_similarity(vectors))
    for i, p in enumerate(processed):                        # sentences with no content words score 0
        if not p:
            scores[i] = 0.0
    return scores


def summarize(text: str, ratio: float = 0.2, max_sentences: int = 0) -> Dict:
    """
    Build an extractive summary.
      ratio         - fraction of sentences to keep (0.05 - 0.6)
      max_sentences - optional hard cap (0 = no cap)
    Returns a dict with the summary, chosen sentence positions and size statistics.
    """
    sentences = split_sentences(text)
    total = len(sentences)
    original_words = len(text.split())
    if total == 0:
        return {"summary": "", "sentences": [], "positions": [], "total_sentences": 0,
                "original_words": original_words, "summary_words": 0, "note": "No sentences found."}

    target = max(1, round(total * ratio))
    if max_sentences > 0:
        target = min(target, max_sentences)
    target = min(target, total)

    if total <= 3 or target >= total:                        # nothing worth cutting
        chosen = list(range(total))
        note = "The document is very short, so the whole text is returned."
    else:
        # Rank each window, then pick the best sentences with a per-window quota.
        windows = [(start, min(start + WINDOW, total)) for start in range(0, total, WINDOW)]
        candidates = []                                      # (score, sentence index)
        for start, end in windows:
            window_sentences = sentences[start:end]
            quota = max(1, round(target * len(window_sentences) / total))
            scores = _rank_window(window_sentences)
            eligible = [i for i, s in enumerate(window_sentences) if len(s.split()) >= MIN_WORDS]
            ranked = sorted(eligible, key=lambda i: scores[i], reverse=True)[:quota]
            candidates.extend((float(scores[i]), start + i) for i in ranked)
        # Quotas are rounded per window, so trim any overshoot by dropping the lowest scores.
        candidates.sort(reverse=True)
        chosen = [index for _, index in candidates[:target]]
        note = ""

    chosen = sorted(set(chosen))                             # restore original reading order
    picked = [sentences[i] for i in chosen]
    summary = " ".join(picked)
    return {
        "summary": summary,
        "sentences": picked,
        "positions": chosen,
        "total_sentences": total,
        "original_words": original_words,
        "summary_words": len(summary.split()),
        "note": note,
    }
