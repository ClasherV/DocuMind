"""
Text preprocessing: turning messy extracted text into clean units a model can use.

Two different "views" of the text are needed:
  * For ML models (classifier, keyword counting): lowercase words only, stop-words removed.
  * For humans (summaries, search results): real sentences / passages with original wording.

Design note: we deliberately use only regular expressions and scikit-learn's built-in
English stop-word list. NLTK/spaCy need extra data downloads at first run (which can fail
on a locked-down Windows PC); for TF-IDF features the benefit of stemming is small.
"""
import re
from typing import List

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOP_WORDS = frozenset(ENGLISH_STOP_WORDS)

# Abbreviations whose trailing "." must NOT end a sentence.
_ABBREVIATIONS = ["dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e",
                  "no", "fig", "inc", "ltd", "co", "approx", "dept", "sec", "cl", "art"]
_ABBR_PATTERN = re.compile(r"\b(" + "|".join(re.escape(a) for a in _ABBREVIATIONS) + r")\.", re.IGNORECASE)
_DOT_PLACEHOLDER = "\u2024"   # a look-alike dot that the sentence regex ignores


def clean_text(text: str) -> str:
    """Normalise whitespace and remove invisible control characters. Keeps paragraph breaks."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)   # control characters
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)                      # re-join hyphenated words
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    """Lowercase and split into alphabetic words (keeps apostrophes inside words)."""
    return re.findall(r"[a-z][a-z']+", text.lower())


def remove_stop_words(tokens: List[str]) -> List[str]:
    """Drop very common words ('the', 'is', 'of') that carry little topic information."""
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 2]


def preprocess_for_model(text: str) -> str:
    """
    Full ML preprocessing used by the classifier AND the TF-IDF search:
    lowercase -> keep words -> remove stop-words -> join back into a string.
    (Module-level so scikit-learn can pickle it together with the trained model.)
    """
    tokens = remove_stop_words(tokenize(text))
    return " ".join(tokens)


def split_paragraphs(text: str) -> List[str]:
    """Split on blank lines; fall back to line groups for text that has none (typical for OCR)."""
    text = clean_text(text)
    if not text:
        return []
    paragraphs = [re.sub(r"\s*\n\s*", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    paragraphs = [p for p in paragraphs if p]
    return paragraphs


def split_sentences(text: str) -> List[str]:
    """
    Rule-based sentence splitter. A sentence ends at . ! or ? followed by whitespace and a
    capital letter / digit / opening quote. Common abbreviations (Dr., e.g., No.) are protected.
    Paragraph boundaries always end a sentence, so headings do not glue onto the next sentence.
    """
    sentences: List[str] = []
    for paragraph in split_paragraphs(text):
        protected = _ABBR_PATTERN.sub(lambda m: m.group(1) + _DOT_PLACEHOLDER, paragraph)
        protected = re.sub(r"(?<=\d)\.(?=\d)", _DOT_PLACEHOLDER, protected)   # 3.5, 75.5%
        parts = re.split(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])", protected)
        for part in parts:
            sentence = part.replace(_DOT_PLACEHOLDER, ".").strip()
            if sentence:
                sentences.append(sentence)
    return sentences


def make_chunks(text: str, max_words: int = 45, overlap_sentences: int = 1) -> List[str]:
    """
    Cut a document into overlapping passages (~max_words each) for semantic search.
    Why chunk? Searching whole documents returns "the document"; searching passages
    returns the exact paragraph that answers the question. Overlap keeps an answer from
    being cut in half at a chunk boundary.
    """
    sentences = split_sentences(text)
    chunks: List[str] = []
    current: List[str] = []
    current_words = 0
    for sentence in sentences:
        n_words = len(sentence.split())
        if current and current_words + n_words > max_words:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
            current_words = sum(len(s.split()) for s in current)
        current.append(sentence)
        current_words += n_words
    if current:
        last = " ".join(current)
        if not chunks or last != chunks[-1]:
            chunks.append(last)
    return chunks
