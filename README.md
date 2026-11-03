# DocuMind — Intelligent Document Understanding System

**Category:** Artificial Intelligence · Machine Learning · NLP · OCR  
**Interface:** Streamlit dashboard (dark navy theme, blue/cyan accents)

DocuMind lets you upload a PDF or an image and then **extract its text, classify its topic, summarise it, search it with natural-language questions, and view statistics** about it. Everything runs **locally** — no paid API keys, and your documents are never sent to an external service.

---

## 1. Problem Statement

People deal with long PDFs, scanned forms and photographed pages. Reading them in full to find one rule, or to decide what kind of document they hold, is slow. Scanned documents are worse: they are pictures, so their text cannot be searched or copied.

## 2. Objectives

1. Read text from digital PDFs and (via OCR) from scans and images.
2. Classify documents into Education, Business, Finance, Technology, Legal or General with a real ML model.
3. Produce shorter extractive summaries with user-controlled length.
4. Find the passages that answer a natural-language question.
5. Show clear document statistics and charts.
6. Stay beginner-friendly, modular and honest about limitations.

## 3. Features

| Feature | What it does | Technique |
|---|---|---|
| Upload & validation | Accepts PDF/JPG/JPEG/PNG, checks extension, size and real file signature | Python checks |
| Text extraction | Reads the PDF text layer; falls back to OCR for scanned pages; OCR for images | PyMuPDF, Tesseract |
| Classification | Predicts category + confidence + the words that influenced it | TF-IDF + Logistic Regression / Naive Bayes |
| Summarisation | Picks the most central sentences; length is adjustable | TextRank over TF-IDF vectors |
| Search | Returns the best-matching passages for a question | Sentence-Transformer embeddings, or TF-IDF fallback |
| Insights | Words, characters, paragraphs, sentences, keywords, charts, processing status | Counting + Matplotlib/Seaborn |

## 4. Technology Stack

Python · Streamlit · Pandas · NumPy · scikit-learn · PyMuPDF · Tesseract OCR (pytesseract) · Pillow · Sentence-Transformers *(optional)* · Matplotlib · Seaborn · joblib

> **NLTK/spaCy:** the brief allows either. DocuMind uses neither on purpose: NLTK needs extra data downloads on first run (which can fail on locked-down PCs) and spaCy needs a separate model download. Instead, tokenisation and sentence splitting are done with regular expressions and scikit-learn's built-in English stop-word list (see `modules/text_preprocessor.py`). If you want stemming or lemmatisation, that is the one file to extend.

## 5. System Architecture

```text
DocuMind/
├── app.py                      Streamlit interface (7 pages) - UI only
├── requirements.txt            core dependencies
├── requirements-semantic.txt   optional: sentence-transformers
├── requirements-dev.txt        optional: reportlab (to regenerate demo files)
├── .streamlit/config.toml      dark navy theme
├── modules/
│   ├── extraction_result.py    shared result containers
│   ├── pdf_processor.py        PDF text extraction + scanned-page detection
│   ├── ocr_processor.py        Tesseract OCR, image validation
│   ├── text_preprocessor.py    cleaning, tokens, sentences, paragraphs, chunks
│   ├── classifier.py           train / save / load / predict / explain
│   ├── summarizer.py           extractive TextRank summariser
│   ├── semantic_search.py      embeddings search + TF-IDF fallback
│   └── document_insights.py    statistics and charts
├── models/
│   ├── classifier.pkl          trained pipeline (auto-retrained if incompatible)
│   └── metrics.json            REAL held-out evaluation written at training time
├── data/
│   ├── training_data.csv       SYNTHETIC demo dataset (630 documents)
│   └── sample_documents/       DEMO PDFs/images + invalid/ broken files for testing
├── scripts/
│   ├── generate_demo_dataset.py
│   └── create_sample_documents.py
├── tests/test_pipeline.py      automated tests
└── assets/style.css            dashboard styling
```

## 6. Machine Learning & NLP Workflow

```text
Upload → Text Extraction → Preprocessing → TF-IDF Features → Classification
                                   ↓
                     Summarisation · Semantic Search · Insights
```

1. **Upload** — validated by extension, size and file signature ("magic bytes").
2. **Extraction** — PyMuPDF reads each PDF page's text layer. If a page has almost no text it is treated as a scan: it is rendered to an image and OCR'd with Tesseract.
3. **Preprocessing** — clean whitespace, lowercase, keep alphabetic words, remove stop-words. Separately, the text is split into paragraphs and sentences for humans to read.
4. **Feature extraction (TF-IDF)** — each word gets a weight: high if frequent in *this* document, low if it appears in *every* document. Single words and two-word phrases are used.
5. **Classification** — Logistic Regression and Multinomial Naive Bayes are both trained and evaluated on a held-out split; the better macro-F1 wins (ties → Logistic Regression), then it is refit on all data.
6. **Summarisation** — sentences become TF-IDF vectors; cosine similarity links them in a graph; PageRank (TextRank) scores each sentence; top sentences are returned in original order. Long documents are processed in windows of 150 sentences.
7. **Search** — the document is cut into ~45-word overlapping passages. Passages and the query are embedded (semantic) or TF-IDF-vectorised (keyword) and ranked by cosine similarity.
8. **Insights** — counts, keyword frequencies and charts.

### Semantic vs. keyword search

| | Keyword (TF-IDF) | Semantic (embeddings) |
|---|---|---|
| Compares | words | meaning |
| "minimum presence needed" finds "attendance rule" | ❌ no shared words | ✅ likely |
| "plagiarise" finds "plagiarism" | ❌ different word forms | ✅ likely |
| Install | built in | `pip install -r requirements-semantic.txt` (large) |
| Network | none | one-time ~90 MB model download from huggingface.co (document text never leaves your PC) |

## 7. Installation on Windows

**Prerequisites:** Windows 10/11 and **Python 3.10 – 3.12** from python.org (tick *"Add Python to PATH"* during setup).

### 7.1 Install Tesseract OCR (a separate program)

1. Download the Windows installer from the UB Mannheim build page: <https://github.com/UB-Mannheim/tesseract/wiki>.
2. Run it with the default folder: `C:\Program Files\Tesseract-OCR`. Keep **English** language data selected.
3. DocuMind finds Tesseract automatically in these places (in order): the `TESSERACT_CMD` environment variable → your `PATH` → the default Windows install folders.
4. Check it works (open a *new* terminal): 
   ```bat
   "C:\Program Files\Tesseract-OCR\tesseract.exe" --version
   ```
5. Only if DocuMind says *"Tesseract OCR was not found"*, set the path once and reopen the terminal:
   ```bat
   setx TESSERACT_CMD "C:\Program Files\Tesseract-OCR\tesseract.exe"
   ```

### 7.2 Create the environment and install dependencies

Open **Command Prompt** (recommended) in the project folder:

```bat
cd path\to\DocuMind
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

*(PowerShell users: if activation is blocked, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first, then `venv\Scripts\Activate.ps1`.)*

Optional — enable true semantic search (large download, includes PyTorch):

```bat
pip install -r requirements-semantic.txt
```

### 7.3 Run

```bat
streamlit run app.py
```

Your browser opens at `http://localhost:8501`. Go to **Document Upload** and click **Load demo document** for an instant test.

### 7.4 Retrain the classifier (optional)

```bat
python -m modules.classifier
```

This retrains from `data/training_data.csv` and rewrites `models/classifier.pkl` and `models/metrics.json`. The app also retrains automatically if the saved model is missing or was saved by a different scikit-learn version.

## 8. Testing

```bat
python -m unittest discover -s tests -v
```

Covered: digital PDF extraction, scanned-PDF OCR fallback, image OCR, classification (per-category texts and the demo documents), summarisation (length control, long documents, empty text), TF-IDF and semantic search, document statistics, and invalid files (empty, corrupted, fake extension, unsupported type, blank image).

Tests that need an optional component (**PyMuPDF**, **Tesseract**, **sentence-transformers**) are **skipped automatically — not faked — when it is missing**. Run the suite on your own machine with everything installed and confirm that nothing is skipped.

### Demo data vs. real documents

| File | What it is |
|---|---|
| `data/training_data.csv` | **Synthetic.** 630 documents assembled from hand-written sentences. Train and test splits use different sentence pools, but the vocabulary is much cleaner than real documents, so the accuracy in `metrics.json` is **optimistic**. |
| `data/sample_documents/demo_*.pdf / .png` | **Fictional demo documents** (academic regulations, finance report, technology note, legal agreement image, scanned business memo). Scans are clean computer-generated images, easier than real scans. |
| `data/sample_documents/invalid/` | Deliberately broken files for error-handling tests. |
| Anything you upload | Real user document, processed locally. |

Suggested manual checks: load `demo_academic_regulations.pdf` and ask *"What is the attendance requirement?"*; load `demo_legal_agreement.png` to see OCR; load `demo_scanned_business_memo.pdf` to see the per-page OCR fallback; use *Test error handling* on the Upload page.

## 9. Known Limitations

- **OCR errors:** blurry, skewed, handwritten, low-contrast or multi-column images produce mistakes. Tesseract's confidence score is only a rough hint. Only English is configured.
- **Scanned PDFs are slow** (each page is rendered and OCR'd) and capped at 60 pages.
- **Classification:** six broad categories, small synthetic training set, no "none of the above" class beyond *General*. Business vs. Finance overlap heavily. Confident predictions can still be wrong. Replace `training_data.csv` with real labelled documents for real use.
- **Summaries are extractive**, not generative: sentences are copied, not rewritten, and can read choppily. Tables, bullet lists and OCR noise reduce quality.
- **Keyword search** only matches words (no stemming/synonyms). **Semantic search** can still return a wrong passage and needs the optional install. Neither *answers* questions; both return passages.
- Sentence/paragraph splitting is rule-based and can be wrong on unusual formatting.
- Password-protected PDFs are rejected. Layout, tables, images and formulas inside PDFs are not interpreted.
- Pickled models depend on the scikit-learn version, which is why the app retrains automatically on a mismatch.

## 10. Privacy

Documents are read into memory and processed on your computer. DocuMind makes **no network calls** except the optional one-time model download for semantic search (the model comes *to* you; your text is not uploaded). Your document's contents are not written to disk by the app; only files you explicitly download (extracted text, summary) are saved, plus the model cache if you enable semantic search.

## 11. Extending the Project

Add categories (edit the CSV + retrain) · add stemming in `text_preprocessor.py` · add a language pack for Tesseract · compare more classifiers · evaluate with real labelled data and report cross-validated scores.
