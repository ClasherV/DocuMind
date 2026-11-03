"""
Search inside one document with a natural-language question.

Two interchangeable back-ends:

1. "semantic"  - Sentence Transformers (model: all-MiniLM-L6-v2). Every passage and the query
                 are converted to embeddings (vectors of ~384 numbers that capture MEANING).
                 Passages whose vectors point in a similar direction (cosine similarity) are
                 returned, so "minimum presence needed" can match "attendance requirement"
                 even though they share no words.
                 PRIVACY NOTE: the first use downloads the model (~90 MB) from huggingface.co.
                 Only the model is downloaded; your document text is never sent anywhere -
                 the embeddings are computed on your own computer. After the first download
                 it works offline.

2. "tfidf"     - Classic keyword retrieval. Passages and query become TF-IDF vectors and are
                 compared with cosine similarity. Fast and needs no download, but it only
                 matches WORDS: a query using different vocabulary than the document finds
                 nothing. This is the automatic fallback.
"""
from functools import lru_cache
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from modules.text_preprocessor import make_chunks, preprocess_for_model

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class SearchBackendError(RuntimeError):
    """Raised when the requested back-end cannot be used (e.g. package missing, offline)."""


def is_semantic_available() -> bool:
    """True if the sentence-transformers package can be imported (model may still need a download)."""
    try:
        import sentence_transformers  # noqa: F401
        return True
    except Exception:  # noqa: BLE001 - ImportError or a broken torch install
        return False


@lru_cache(maxsize=1)
def _load_embedding_model():
    """Load (and on first use download) the embedding model. Cached for the whole session."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(MODEL_NAME)
    except ImportError as exc:
        raise SearchBackendError(
            "sentence-transformers is not installed. Run: pip install -r requirements-semantic.txt"
        ) from exc
    except Exception as exc:  # noqa: BLE001 - network failure, disk problem, ...
        raise SearchBackendError(
            f"The embedding model could not be loaded (internet needed for the first download): {exc}"
        ) from exc


class DocumentSearchEngine:
    """Index a document's passages once, then answer many queries quickly."""

    def __init__(self, text: str, backend: str = "tfidf", max_words: int = 45):
        self.chunks: List[str] = make_chunks(text, max_words=max_words)
        if not self.chunks:
            raise ValueError("There is no text to index.")
        self.backend = backend

        if backend == "semantic":
            model = _load_embedding_model()
            self._embeddings = model.encode(self.chunks, normalize_embeddings=True,
                                            show_progress_bar=False, batch_size=32)
        elif backend == "tfidf":
            self._vectorizer = TfidfVectorizer(preprocessor=preprocess_for_model,
                                               ngram_range=(1, 2), sublinear_tf=True)
            try:
                self._matrix = self._vectorizer.fit_transform(self.chunks)
            except ValueError as exc:   # no usable words at all
                raise ValueError("The document has no searchable words.") from exc
        else:
            raise ValueError(f"Unknown search backend: {backend}")

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Return the top_k most relevant passages as dicts: rank, score (0-1), text, chunk_index."""
        query = (query or "").strip()
        if not query:
            return []

        if self.backend == "semantic":
            query_vec = _load_embedding_model().encode([query], normalize_embeddings=True,
                                                       show_progress_bar=False)
            scores = (self._embeddings @ query_vec[0]).astype(float)
        else:
            query_vec = self._vectorizer.transform([query])
            if query_vec.nnz == 0:        # none of the query words exist in the document
                return []
            scores = cosine_similarity(query_vec, self._matrix)[0]

        order = np.argsort(scores)[::-1][:top_k]
        results = []
        for rank, idx in enumerate(order, start=1):
            if self.backend == "tfidf" and scores[idx] <= 0:
                continue                   # a zero score means no shared words - not a real match
            results.append({"rank": rank, "score": float(max(scores[idx], 0.0)),
                            "text": self.chunks[idx], "chunk_index": int(idx)})
        return results
