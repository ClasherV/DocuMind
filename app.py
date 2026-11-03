"""
DocuMind - Intelligent Document Understanding System (Streamlit front-end).

Run with:   streamlit run app.py

This file only handles the user interface. All the real work (PDF reading, OCR,
classification, summarisation, search, statistics) lives in the `modules/` package.
Everything is processed locally on your computer; documents are never uploaded anywhere.
"""
import hashlib
import html
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from modules import classifier, ocr_processor, pdf_processor, summarizer
from modules import document_insights as insights_lib
from modules.semantic_search import DocumentSearchEngine, SearchBackendError, is_semantic_available
from modules.text_preprocessor import clean_text, preprocess_for_model

ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "data" / "sample_documents"
INVALID_DIR = SAMPLE_DIR / "invalid"
ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png"}
MAX_UPLOAD_MB = 25

PAGES = ["Home", "Document Upload", "Text Extraction", "Document Classification",
         "Text Summarization", "Semantic Search", "Document Insights"]

st.set_page_config(page_title="DocuMind", page_icon="📄", layout="wide")


# --------------------------------------------------------------------------- styling & helpers
def load_css() -> None:
    css_file = ROOT / "assets" / "style.css"
    if css_file.exists():
        st.markdown(f"<style>{css_file.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def page_header(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="dm-header"><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div>',
                unsafe_allow_html=True)


def metric_card(label: str, value, sub: str = "", accent: bool = False) -> str:
    cls = "dm-value accent" if accent else "dm-value"
    sub_html = f'<div class="dm-sub">{html.escape(sub)}</div>' if sub else ""
    return (f'<div class="dm-card"><div class="dm-label">{html.escape(label)}</div>'
            f'<div class="{cls}">{html.escape(str(value))}</div>{sub_html}</div>')


def metric_row(cards) -> None:
    """cards = list of (label, value, sub, accent) tuples shown side by side."""
    columns = st.columns(len(cards))
    for col, card in zip(columns, cards):
        with col:
            st.markdown(metric_card(*card), unsafe_allow_html=True)


def show_figure(fig) -> None:
    if fig is not None:
        st.pyplot(fig)
        plt.close(fig)


def go_to(page: str) -> None:
    """Button callback: switch the sidebar navigation to another page."""
    st.session_state["page"] = page


@st.cache_resource(show_spinner="Loading classification model...")
def get_classifier_bundle():
    return classifier.load_or_train()


@st.cache_resource
def get_tesseract_status():
    return ocr_processor.configure_tesseract()


@st.cache_data(show_spinner=False)
def run_summary(text: str, ratio: float, max_sentences: int):
    return summarizer.summarize(text, ratio=ratio, max_sentences=max_sentences)


def init_state() -> None:
    defaults = {"page": "Home", "doc": None, "last_upload_hash": None, "engines": {},
                "summary_ran": False, "search_ran": False}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


# --------------------------------------------------------------------------- upload & processing
def validate_upload(name: str, data: bytes):
    """Return an error message, or None if the file looks acceptable."""
    extension = Path(name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        return f"Unsupported file type '{extension or 'none'}'. Please upload a PDF, JPG, JPEG or PNG file."
    if not data:
        return "The file is empty (0 bytes)."
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        return f"The file is larger than {MAX_UPLOAD_MB} MB."
    # Check the real file signature, not just the extension.
    if extension == ".pdf" and not data.lstrip()[:5] == b"%PDF-":
        return "The file has a .pdf extension but is not a real PDF."
    if extension == ".png" and not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "The file has a .png extension but is not a real PNG image."
    if extension in (".jpg", ".jpeg") and not data.startswith(b"\xff\xd8"):
        return "The file has a .jpg extension but is not a real JPEG image."
    return None


def process_document(name: str, data: bytes, is_demo: bool = False):
    """
    Run validation -> extraction (PDF text layer or OCR) -> cleaning -> classification -> statistics.
    Returns (doc_dict, None) on success or (None, error_message) on failure.
    """
    error = validate_upload(name, data)
    if error:
        return None, error

    extension = Path(name).suffix.lower()
    with st.spinner("Extracting text (scanned pages and images use OCR and can take a few seconds)..."):
        if extension == ".pdf":
            result = pdf_processor.extract_text_from_pdf(data)
        else:
            result = ocr_processor.extract_text_from_image(data)
    if not result.ok:
        return None, result.error

    text = clean_text(result.text)
    if len(text.split()) < 3:
        return None, "Almost no text was found in this document, so there is nothing to analyse."

    with st.spinner("Classifying and computing statistics..."):
        classification = classifier.predict(get_classifier_bundle(), text)
        stats = insights_lib.compute_insights(text)

    doc = {"name": name, "is_demo": is_demo, "hash": hashlib.sha256(data).hexdigest(),
           "size_kb": len(data) / 1024, "extraction": result, "text": text,
           "classification": classification, "insights": stats}
    return doc, None


def set_document(doc: dict) -> None:
    """Make `doc` the active document and reset everything derived from the previous one."""
    st.session_state["doc"] = doc
    st.session_state["engines"] = {}
    st.session_state["summary_ran"] = False
    st.session_state["search_ran"] = False


def require_document():
    """Return the active document, or show a friendly hint and return None."""
    doc = st.session_state.get("doc")
    if doc is None:
        st.info("No document loaded yet. Upload one (or load a demo document) first.")
        st.button("Go to Document Upload", on_click=go_to, args=("Document Upload",))
    return doc


# --------------------------------------------------------------------------- sidebar
def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 📄 DocuMind")
        st.caption("Intelligent Document Understanding")
        st.radio("Navigate", PAGES, key="page", label_visibility="collapsed")
        st.divider()

        doc = st.session_state.get("doc")
        if doc:
            tag = "DEMO" if doc["is_demo"] else "Your file"
            st.markdown(f"**Active document**  \n`{doc['name']}`  \n{tag} · {doc['insights']['words']:,} words")
            if st.button("Clear document"):
                st.session_state["doc"] = None
                st.session_state["engines"] = {}
                st.rerun()
        else:
            st.caption("No document loaded.")

        st.divider()
        ocr_ok, _ = get_tesseract_status()
        pill = lambda ok, text: f'<span class="dm-pill {"ok" if ok else "warn"}">{text}</span>'
        st.markdown(pill(ocr_ok, "OCR ready" if ocr_ok else "OCR unavailable")
                    + pill(is_semantic_available(), "Semantic search ready" if is_semantic_available()
                           else "Keyword search only"), unsafe_allow_html=True)
        st.caption("🔒 Documents are processed locally on this computer.")


# --------------------------------------------------------------------------- pages
def page_home() -> None:
    st.markdown(
        '<div class="dm-hero"><h1>Docu<span>Mind</span></h1>'
        "<p>Upload a PDF or image and explore it: extract text, classify the topic, summarise it, "
        "search it with natural-language questions and see document statistics.</p></div>",
        unsafe_allow_html=True)

    st.markdown("#### How the pipeline works")
    steps = ["Upload", "Text extraction (PDF / OCR)", "Preprocessing", "TF-IDF features",
             "Classification", "Summarisation", "Search", "Insights"]
    st.markdown('<div class="dm-pipeline">' + '<span class="dm-arrow">→</span>'.join(
        f'<span class="dm-step">{s}</span>' for s in steps) + "</div>", unsafe_allow_html=True)

    features = [
        ("Text extraction", "Reads digital PDFs with PyMuPDF and scanned pages or images with Tesseract OCR."),
        ("Classification", "TF-IDF + a scikit-learn model assigns a category and shows its confidence."),
        ("Summarisation", "Extractive TextRank picks the most central sentences. You control the length."),
        ("Semantic search", "Ask questions in plain English and get the matching passages."),
        ("Insights", "Word counts, sentences, keywords and charts that describe the document."),
        ("Local & private", "No paid API keys. Your documents never leave your computer."),
    ]
    for row in (features[:3], features[3:]):
        for col, (title, body) in zip(st.columns(3), row):
            with col:
                st.markdown(f'<div class="dm-card"><div class="dm-label">{title}</div>'
                            f'<div class="dm-sub" style="font-size:.92rem;color:#B6C4DB;margin-top:6px">{body}</div></div>',
                            unsafe_allow_html=True)
        st.write("")

    st.button("Get started: upload a document →", on_click=go_to, args=("Document Upload",), type="primary")

    with st.expander("Honest limitations (please read)"):
        st.markdown(
            "- **OCR makes mistakes**, especially on blurry, skewed, handwritten or low-contrast images.\n"
            "- **Classification** only knows six broad categories and was trained on a small dataset. "
            "If you use the bundled *synthetic demo dataset*, its accuracy figure is optimistic and says "
            "little about real documents.\n"
            "- **Summaries are extractive**: they reuse original sentences. This is not generative AI, "
            "so it can read choppily and miss ideas spread across many sentences.\n"
            "- **Keyword search** only matches words; **semantic search** matches meaning but can still return "
            "a wrong passage. Always read the source text.\n"
            "- Tuned for **English** text.")


def page_upload() -> None:
    page_header("Document Upload", "Upload a PDF, JPG, JPEG or PNG file, or try a bundled demo document.")
    st.info("🔒 Your file is read into memory and processed on this computer. It is not sent to any external "
            "service. (The only optional download is the embedding model for semantic search.)")

    uploaded = st.file_uploader("Choose a file", type=["pdf", "jpg", "jpeg", "png"],
                                help=f"Maximum {MAX_UPLOAD_MB} MB")
    if uploaded is not None:
        data = uploaded.getvalue()
        digest = hashlib.sha256(data).hexdigest()
        if digest != st.session_state["last_upload_hash"]:       # process each new upload only once
            st.session_state["last_upload_hash"] = digest
            doc, error = process_document(uploaded.name, data)
            if error:
                st.error(f"Could not process **{uploaded.name}**: {error}")
            else:
                set_document(doc)
                st.success(f"Processed **{uploaded.name}** successfully.")
        elif st.session_state["doc"] is not None and st.session_state["doc"]["hash"] == digest:
            st.success(f"**{uploaded.name}** is loaded.")

    st.markdown("#### Or try a demo document")
    st.caption("These files are fictional DEMO documents written for testing - not real user documents.")
    demo_files = sorted(p for p in SAMPLE_DIR.glob("*") if p.suffix.lower() in ALLOWED_EXTENSIONS) \
        if SAMPLE_DIR.exists() else []
    if demo_files:
        choice = st.selectbox("Demo document", demo_files, format_func=lambda p: p.name)
        if st.button("Load demo document"):
            doc, error = process_document(choice.name, choice.read_bytes(), is_demo=True)
            if error:
                st.error(f"Could not process the demo file: {error}")
            else:
                set_document(doc)
                st.session_state["last_upload_hash"] = None
                st.success(f"Loaded DEMO document **{choice.name}**.")
    else:
        st.warning("No demo documents found in data/sample_documents/.")

    with st.expander("Test error handling with deliberately broken files"):
        bad_files = sorted(INVALID_DIR.glob("*")) if INVALID_DIR.exists() else []
        if bad_files:
            bad = st.selectbox("Invalid test file", bad_files, format_func=lambda p: p.name)
            if st.button("Try processing this file"):
                _, error = process_document(bad.name, bad.read_bytes(), is_demo=True)
                if error:
                    st.warning(f"Handled gracefully - no crash. Message shown to the user: {error}")
                else:
                    st.success("This file was processed without problems.")

    doc = st.session_state["doc"]
    if doc:
        st.markdown("#### Current document")
        ext = doc["extraction"]
        metric_row([("File", doc["name"], "DEMO file" if doc["is_demo"] else "Uploaded", False),
                    ("Type", ext.source_type.upper(), f"{doc['size_kb']:.0f} KB", False),
                    ("Extraction", ext.methods_used, f"{len(ext.pages)} page(s)", True),
                    ("Words", f"{doc['insights']['words']:,}", "", False)])
        for warning in ext.warnings:
            st.warning(warning)
        st.button("Next: view extracted text →", on_click=go_to, args=("Text Extraction",))


def page_extraction() -> None:
    page_header("Text Extraction", "The raw text DocuMind read from your document.")
    doc = require_document()
    if doc is None:
        return
    ext = doc["extraction"]
    conf = ext.mean_ocr_confidence
    metric_row([("Source", ext.source_type.upper(), "", False),
                ("Method", ext.methods_used, "digital = text layer · ocr = Tesseract", True),
                ("Pages", len(ext.pages), "", False),
                ("OCR confidence", f"{conf:.0f}/100" if conf is not None else "n/a",
                 "Tesseract's own estimate" if conf is not None else "no OCR used", False)])
    for warning in ext.warnings:
        st.warning(warning)
    if conf is not None:
        st.caption("OCR text can contain mistakes (e.g. 'rn' read as 'm', 'l' read as '1'). "
                   "Downstream results are only as good as the extracted text.")

    full_tab, pages_tab = st.tabs(["Full text", "Page by page"])
    with full_tab:
        st.text_area("Extracted text", doc["text"], height=420, disabled=True, label_visibility="collapsed")
        st.download_button("Download text (.txt)", doc["text"].encode("utf-8"),
                           file_name=f"{Path(doc['name']).stem}_extracted.txt", mime="text/plain")
    with pages_tab:
        for page in ext.pages:
            label = f"Page {page.page_number} · {page.method}"
            if page.ocr_confidence is not None:
                label += f" · OCR confidence {page.ocr_confidence:.0f}"
            with st.expander(label):
                st.text(page.text if page.text.strip() else "(no text on this page)")


def page_classification() -> None:
    page_header("Document Classification", "A machine-learning model predicts the document's topic.")
    doc = require_document()
    if doc is None:
        return
    result = doc["classification"]

    if result["insufficient_text"]:
        st.warning(result["message"])
    else:
        left, right = st.columns([1, 1])
        with left:
            st.markdown(
                f'<div class="dm-result"><div class="dm-label">Predicted category</div>'
                f'<div class="dm-value accent" style="font-size:2.4rem">{html.escape(result["label"])}</div>'
                f'<div class="dm-sub">Confidence: {result["confidence"] * 100:.1f}%</div></div>',
                unsafe_allow_html=True)
            if result["low_confidence"]:
                st.warning("Low confidence: the model is not sure. The document may fit several categories "
                           "or none of them well.")
        with right:
            show_figure(insights_lib.probability_chart(result["probabilities"]))
        show_figure(insights_lib.term_weight_chart(result["top_terms"]))

    st.caption("Confidence is the model's own probability estimate. It is not a guarantee of correctness - "
               "a model can be confidently wrong, especially on topics unlike its training data.")

    with st.expander("How does the classification work?"):
        st.markdown(
            "1. **Preprocessing** - the text is lowercased and common words such as *the* or *is* "
            "(stop-words) are removed.\n"
            "2. **TF-IDF features** - every word becomes a number. A word scores high if it is frequent in "
            "*this* document but rare across *all* training documents, so *attendance* is informative while "
            "*document* is not. Two-word phrases are included too.\n"
            "3. **Model** - a classifier (Logistic Regression or Multinomial Naive Bayes, whichever scored "
            "better on held-out data) learned which words go with which category from labelled examples.\n"
            "4. **Prediction** - the model outputs a probability for each category; the highest wins. "
            "The bar chart above shows those probabilities and the second chart shows which words "
            "pushed the decision.\n\n"
            "The 'General' category is a catch-all. Documents about topics outside the six categories "
            "will usually land somewhere that looks plausible but is not truly right.")

    with st.expander("Model details and real evaluation results"):
        metrics = classifier.load_metrics()
        if not metrics:
            st.info("No evaluation file found. Click 'Retrain model' below.")
        else:
            selected = metrics["selected_algorithm"]
            st.markdown(f"**Selected algorithm:** `{selected}` · trained on **{metrics['n_train']}** documents, "
                        f"tested on **{metrics['n_test']}** held-out documents · "
                        f"data file: `{metrics['dataset_file']}` · trained {metrics['trained_at']}")
            table = pd.DataFrame([{"Algorithm": name, "Accuracy": round(res["accuracy"], 3),
                                   "Macro F1": round(res["macro_f1"], 3)}
                                  for name, res in metrics["held_out_results"].items()])
            st.dataframe(table, hide_index=True)
            show_figure(insights_lib.confusion_matrix_chart(
                metrics["held_out_results"][selected]["confusion_matrix"], metrics["labels"]))
            st.warning(metrics["note"])
        st.caption("To use your own labelled data, replace data/training_data.csv (columns: text, label, split) "
                   "and retrain.")
        if st.button("Retrain model from data/training_data.csv"):
            with st.spinner("Training..."):
                try:
                    classifier.train_and_save()
                    get_classifier_bundle.clear()
                    doc["classification"] = classifier.predict(get_classifier_bundle(), doc["text"])
                    st.success("Model retrained and saved. Reopen this page to see the new results.")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Training failed: {exc}")


def page_summarization() -> None:
    page_header("Text Summarization", "Extractive summary: the most central sentences of the document.")
    doc = require_document()
    if doc is None:
        return

    col1, col2 = st.columns(2)
    with col1:
        percent = st.slider("Summary length (% of sentences)", 5, 60, 20, step=5)
    with col2:
        max_sentences = st.number_input("Maximum sentences (0 = no limit)", min_value=0, max_value=100, value=0)

    result = run_summary(doc["text"], percent / 100, int(max_sentences))
    st.session_state["summary_ran"] = True
    if not result["summary"]:
        st.warning(result["note"] or "No summary could be produced.")
        return

    reduction = 100 * (1 - result["summary_words"] / max(result["original_words"], 1))
    metric_row([("Sentences kept", f"{len(result['sentences'])} / {result['total_sentences']}", "", False),
                ("Original words", f"{result['original_words']:,}", "", False),
                ("Summary words", f"{result['summary_words']:,}", "", False),
                ("Shorter by", f"{reduction:.0f}%", "", True)])
    if result["note"]:
        st.info(result["note"])

    st.markdown(f'<div class="dm-summary">{html.escape(result["summary"])}</div>', unsafe_allow_html=True)
    st.download_button("Download summary (.txt)", result["summary"].encode("utf-8"),
                       file_name=f"{Path(doc['name']).stem}_summary.txt", mime="text/plain")

    with st.expander("Which sentences were chosen?"):
        for position, sentence in zip(result["positions"], result["sentences"]):
            st.markdown(f"**#{position + 1}** · {sentence}")

    st.caption("This is *extractive* summarisation (TextRank). It copies existing sentences, so it cannot "
               "rephrase, combine ideas or explain. Tables, lists and OCR errors can make it pick odd sentences.")


def highlight_terms(text: str, query: str) -> str:
    """HTML-escape a passage and wrap words that also appear in the query with <mark>."""
    safe = html.escape(text)
    tokens = sorted(set(preprocess_for_model(query).split()), key=len, reverse=True)
    if not tokens:
        return safe
    pattern = re.compile(r"\b(" + "|".join(re.escape(t) for t in tokens) + r")\b", re.IGNORECASE)
    return pattern.sub(r"<mark>\1</mark>", safe)


def page_search() -> None:
    page_header("Semantic Search", "Ask a question in plain English and get the matching passages.")
    doc = require_document()
    if doc is None:
        return

    semantic_ok = is_semantic_available()
    options = ["Semantic (Sentence Transformers)", "Keyword (TF-IDF)"] if semantic_ok else ["Keyword (TF-IDF)"]
    backend_label = st.radio("Search method", options, horizontal=True)
    if not semantic_ok:
        st.info("Semantic embeddings are not installed, so the TF-IDF keyword fallback is used. "
                "To enable semantic search run: `pip install -r requirements-semantic.txt`")
    elif backend_label.startswith("Semantic"):
        st.caption("The first semantic search downloads a ~90 MB model from huggingface.co. "
                   "Only the model is downloaded - your document text never leaves your computer.")

    top_k = st.slider("Number of passages", 1, 8, 3)
    query = st.text_input("Your question", placeholder="e.g. What is the attendance requirement?")
    if not query.strip():
        st.caption("Try a question that the document answers. Search results are passages from the document, "
                   "not generated answers.")
        return

    backend = "semantic" if backend_label.startswith("Semantic") else "tfidf"
    cache_key = (doc["hash"], backend)
    engines = st.session_state["engines"]
    if cache_key not in engines:
        try:
            with st.spinner("Indexing the document" + (" (first run downloads the model)..." if backend == "semantic" else "...")):
                engines[cache_key] = DocumentSearchEngine(doc["text"], backend=backend)
        except SearchBackendError as exc:
            st.warning(f"{exc}\n\nFalling back to keyword search.")
            backend, cache_key = "tfidf", (doc["hash"], "tfidf")
            if cache_key not in engines:
                engines[cache_key] = DocumentSearchEngine(doc["text"], backend="tfidf")
        except ValueError as exc:
            st.error(str(exc))
            return

    engine = engines[cache_key]
    st.session_state["search_ran"] = True
    results = engine.search(query, top_k=top_k)
    st.caption(f"Method used: **{'semantic embeddings' if backend == 'semantic' else 'TF-IDF keywords'}** · "
               f"{len(engine.chunks)} passages indexed")

    if not results:
        st.warning("No passage shares any words with your question. Try different words"
                   + ("" if backend == "semantic" else ", or install semantic search to match by meaning") + ".")
        return

    weak_threshold = 0.25 if backend == "semantic" else 0.08
    if results[0]["score"] < weak_threshold:
        st.warning("The best match is weak - the document may not actually answer this question.")

    for item in results:
        st.markdown(f'<div class="dm-passage"><div class="meta">Result {item["rank"]} · passage '
                    f'{item["chunk_index"] + 1} · similarity {item["score"]:.2f}</div>'
                    f'{highlight_terms(item["text"], query)}</div>', unsafe_allow_html=True)
        st.progress(min(max(item["score"], 0.0), 1.0))

    with st.expander("Semantic search vs. keyword (TF-IDF) search"):
        st.markdown(
            "**Keyword / TF-IDF** compares *words*. A question about *minimum presence* will not find a passage "
            "that says *attendance*, and *plagiarise* will not match *plagiarism* because the words differ.\n\n"
            "**Semantic** search converts text into embeddings (lists of numbers capturing meaning), so passages "
            "with similar meaning are close together even when no words are shared. It needs the extra "
            "`sentence-transformers` package and a one-time model download.\n\n"
            "Similarity scores are *not probabilities* and cannot be compared between the two methods. "
            "Neither method *answers* the question - they find the passages most likely to contain the answer.")


def page_insights() -> None:
    page_header("Document Insights", "Key statistics and keywords describing the document.")
    doc = require_document()
    if doc is None:
        return
    s, c = doc["insights"], doc["classification"]

    metric_row([("Total words", f"{s['words']:,}", "", False),
                ("Characters", f"{s['characters']:,}", f"{s['characters_no_spaces']:,} without spaces", False),
                ("Paragraphs", f"{s['paragraphs']:,}", "", False),
                ("Sentences", f"{s['sentences']:,}", f"avg {s['avg_sentence_length']:.1f} words", False)])
    st.write("")
    metric_row([("Category", c["label"] if not c["insufficient_text"] else "Unknown",
                 f"{c['confidence'] * 100:.0f}% confidence", True),
                ("Reading time", f"{max(s['reading_time_min'], 0.1):.1f} min", "at 200 words/min", False),
                ("Unique words", f"{s['unique_words']:,}", f"diversity {s['lexical_diversity']:.2f}", False),
                ("Avg word length", f"{s['avg_word_length']:.1f}", "characters", False)])

    st.markdown("#### Processing status")
    ext = doc["extraction"]
    status = pd.DataFrame([
        ("File validation", "Done", f"{doc['name']} accepted"),
        ("Text extraction", "Done", f"method: {ext.methods_used}, {len(ext.pages)} page(s)"),
        ("Preprocessing", "Done", "cleaned text, tokens and sentences"),
        ("Classification", "Done" if not c["insufficient_text"] else "Skipped", c["label"]),
        ("Summarization", "Done" if st.session_state["summary_ran"] else "Not run yet", "open the Summarization page"),
        ("Semantic search", "Done" if st.session_state["search_ran"] else "Not used yet", "open the Semantic Search page"),
        ("Insights", "Done", "this page"),
    ], columns=["Stage", "Status", "Detail"])
    st.dataframe(status, hide_index=True)

    left, right = st.columns(2)
    with left:
        show_figure(insights_lib.keyword_chart(s["keywords"]))
    with right:
        show_figure(insights_lib.sentence_length_chart(s["sentence_lengths"]))
    st.caption("Keywords are the most frequent non-stop-words. Frequency shows what a document repeats, "
               "not necessarily what is most important.")


# --------------------------------------------------------------------------- main
def main() -> None:
    init_state()
    load_css()
    render_sidebar()
    pages = {
        "Home": page_home, "Document Upload": page_upload, "Text Extraction": page_extraction,
        "Document Classification": page_classification, "Text Summarization": page_summarization,
        "Semantic Search": page_search, "Document Insights": page_insights,
    }
    pages[st.session_state["page"]]()


main()
