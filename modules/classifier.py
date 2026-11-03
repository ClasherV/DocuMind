"""
Document classification with scikit-learn.

Pipeline (one scikit-learn `Pipeline` object, saved as models/classifier.pkl):
    raw text -> preprocess -> TF-IDF vectoriser -> classifier (Logistic Regression or Naive Bayes)

TF-IDF in one sentence: each word gets a weight that is high when the word appears often in
THIS document but rarely across ALL documents - so "attendance" is informative, "the" is not.

Training also runs a real evaluation on a held-out test split and stores the numbers in
models/metrics.json. Nothing in this file is hard-coded or faked.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from modules.text_preprocessor import preprocess_for_model

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "training_data.csv"
MODEL_PATH = ROOT / "models" / "classifier.pkl"
METRICS_PATH = ROOT / "models" / "metrics.json"

CATEGORIES = ["Education", "Business", "Finance", "Technology", "Legal", "General"]
LOW_CONFIDENCE_THRESHOLD = 0.50   # below this we warn the user
MIN_MEANINGFUL_TOKENS = 5         # fewer informative words than this -> refuse to guess


def _build_pipeline(algorithm: str) -> Pipeline:
    """Create an untrained pipeline for 'logreg' or 'naive_bayes'."""
    vectorizer = TfidfVectorizer(
        preprocessor=preprocess_for_model,   # lowercase, stop-word removal (see text_preprocessor)
        ngram_range=(1, 2),                  # single words AND two-word phrases
        sublinear_tf=True,                   # dampen very frequent words (1 + log tf)
        min_df=2,                            # ignore words seen in only one training document
        max_df=0.9,                          # ignore words that appear almost everywhere
    )
    if algorithm == "naive_bayes":
        model = MultinomialNB(alpha=0.1)
    else:
        model = LogisticRegression(max_iter=2000, C=10.0)
    return Pipeline([("tfidf", vectorizer), ("clf", model)])


def _load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"text", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"{csv_path.name} must contain the columns: text, label (and optionally split).")
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)
    return df


def train_and_save(csv_path: Path = DATA_PATH, model_path: Path = MODEL_PATH) -> Dict:
    """
    Train both algorithms, evaluate on the held-out split, keep the better one
    (ties go to Logistic Regression), refit it on ALL data, and save it.
    Returns the saved bundle.
    """
    df = _load_dataset(Path(csv_path))
    if df["label"].nunique() < 2:
        raise ValueError("The training file needs at least two different labels.")

    # Use the file's own split column if present; otherwise make a stratified 80/20 split.
    if "split" in df.columns and set(df["split"].unique()) >= {"train", "test"}:
        train_df, test_df = df[df["split"] == "train"], df[df["split"] == "test"]
    else:
        train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

    labels = sorted(df["label"].unique())
    comparison = {}
    for algorithm in ("logreg", "naive_bayes"):
        pipe = _build_pipeline(algorithm)
        pipe.fit(train_df["text"], train_df["label"])
        predicted = pipe.predict(test_df["text"])
        comparison[algorithm] = {
            "accuracy": float(accuracy_score(test_df["label"], predicted)),
            "macro_f1": float(f1_score(test_df["label"], predicted, average="macro")),
            "per_class": classification_report(test_df["label"], predicted, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(test_df["label"], predicted, labels=labels).tolist(),
        }

    best = "naive_bayes" if comparison["naive_bayes"]["macro_f1"] > comparison["logreg"]["macro_f1"] else "logreg"
    final_pipeline = _build_pipeline(best)
    final_pipeline.fit(df["text"], df["label"])      # refit on everything for the shipped model

    metrics = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_file": Path(csv_path).name,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "labels": labels,
        "selected_algorithm": best,
        "held_out_results": comparison,
        "note": ("Metrics come from the held-out test split BEFORE the final refit on all data. "
                 "If the dataset is the synthetic demo set, they are optimistic and do not "
                 "predict performance on real documents."),
    }
    bundle = {"pipeline": final_pipeline, "sklearn_version": sklearn.__version__,
              "algorithm": best, "metrics": metrics}

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    with open(METRICS_PATH, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return bundle


def load_or_train(model_path: Path = MODEL_PATH) -> Dict:
    """
    Load the saved model. If it is missing, unreadable, or was saved by a different
    scikit-learn version (pickles are not portable across versions), retrain automatically.
    """
    try:
        bundle = joblib.load(model_path)
        if bundle.get("sklearn_version") == sklearn.__version__:
            return bundle
    except Exception:  # noqa: BLE001 - any load problem -> just retrain
        pass
    return train_and_save(DATA_PATH, model_path)


def _top_terms(pipeline: Pipeline, text: str, predicted_label: str, n_terms: int = 8) -> List[Dict]:
    """Which words pushed the model toward its answer? (a simple explanation)"""
    vectorizer, model = pipeline.named_steps["tfidf"], pipeline.named_steps["clf"]
    x = vectorizer.transform([text])
    if x.nnz == 0:
        return []
    class_index = list(model.classes_).index(predicted_label)
    if hasattr(model, "coef_"):                       # Logistic Regression: learned word weights
        weights = model.coef_[class_index]
    else:                                              # Naive Bayes: log-prob vs. the other classes
        log_prob = model.feature_log_prob_
        weights = log_prob[class_index] - np.delete(log_prob, class_index, axis=0).mean(axis=0)
    contributions = x.multiply(weights).tocoo()
    names = vectorizer.get_feature_names_out()
    ranked = sorted(zip(contributions.col, contributions.data), key=lambda p: p[1], reverse=True)
    return [{"term": names[i], "weight": float(w)} for i, w in ranked[:n_terms] if w > 0]


def predict(bundle: Dict, text: str) -> Dict:
    """Classify one document. Returns label, confidence, all probabilities and explanation."""
    pipeline = bundle["pipeline"]
    tokens = preprocess_for_model(text).split()
    if len(tokens) < MIN_MEANINGFUL_TOKENS:
        return {"label": "General", "confidence": 0.0, "probabilities": {}, "top_terms": [],
                "low_confidence": True, "insufficient_text": True,
                "message": "Too little text to classify reliably - defaulting to 'General'."}

    probabilities = pipeline.predict_proba([text])[0]
    classes = list(pipeline.named_steps["clf"].classes_)
    best = int(np.argmax(probabilities))
    label, confidence = classes[best], float(probabilities[best])
    return {
        "label": label,
        "confidence": confidence,
        "probabilities": {c: float(p) for c, p in sorted(zip(classes, probabilities), key=lambda cp: -cp[1])},
        "top_terms": _top_terms(pipeline, text, label),
        "low_confidence": confidence < LOW_CONFIDENCE_THRESHOLD,
        "insufficient_text": False,
        "message": "",
    }


def load_metrics() -> Optional[Dict]:
    """Return the evaluation metrics saved at training time (or None)."""
    try:
        with open(METRICS_PATH, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


if __name__ == "__main__":
    # `python -m modules.classifier` retrains the model from data/training_data.csv
    saved = train_and_save()
    m = saved["metrics"]
    chosen = m["held_out_results"][m["selected_algorithm"]]
    print(f"Selected: {m['selected_algorithm']}  | held-out accuracy {chosen['accuracy']:.3f}  "
          f"| macro-F1 {chosen['macro_f1']:.3f}  (train={m['n_train']}, test={m['n_test']})")
    for name, res in m["held_out_results"].items():
        print(f"  {name:12s} accuracy={res['accuracy']:.3f} macro_f1={res['macro_f1']:.3f}")
