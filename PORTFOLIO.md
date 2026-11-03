# DocuMind — Portfolio & Viva Guide

## 1. Project description (copy-paste)

> **DocuMind** is a local, privacy-friendly document intelligence app that extracts text from PDFs and scanned images (PyMuPDF + Tesseract OCR), classifies the document topic with a TF-IDF + scikit-learn model, and generates extractive summaries. It also answers natural-language questions by retrieving the most relevant passages using sentence-embedding or TF-IDF search, and presents document statistics in a Streamlit dashboard.

## 2. Technology tags

`Python` `Streamlit` `scikit-learn` `TF-IDF` `Logistic Regression` `Naive Bayes` `NLP` `OCR` `Tesseract` `PyMuPDF` `Sentence-Transformers` `TextRank` `Pandas` `NumPy` `Matplotlib` `Seaborn`

## 3. Key features

- PDF and image upload with validation (extension, size, real file signature)
- Digital-PDF text extraction with automatic **per-page OCR fallback** for scanned pages
- ML document classification into 6 categories, with confidence, influential words and honest evaluation metrics
- Extractive TextRank summarisation with adjustable length, scalable to long documents
- Natural-language passage search: semantic embeddings, with a TF-IDF fallback
- Document insights: words, characters, paragraphs, sentences, keywords, charts, processing status
- Graceful handling of empty, corrupted, fake-extension and blank files
- Runs fully offline on Windows, no API keys; modular code with automated tests

## 4. How to present it to your teacher (≈3 minutes)

1. **Problem (20 s):** "Scanned and long documents are hard to search and skim. DocuMind turns them into searchable, classified, summarised text."
2. **Demo (90 s):**
   - Upload `demo_academic_regulations.pdf` → show *Text Extraction* (method: digital).
   - *Classification*: Education + confidence + the words that drove it. Open the model-details panel and say honestly that the dataset is synthetic.
   - *Summarization*: move the slider and show the summary shrink.
   - *Semantic Search*: ask "What is the attendance requirement?" and show the highlighted passage.
   - Load `demo_scanned_business_memo.pdf` → method: **ocr**, show OCR confidence. Then try a broken file from *Test error handling*.
3. **How it works (45 s):** Show the pipeline on the Home page: extraction → preprocessing → TF-IDF → classifier / TextRank / search → insights. Explain that `app.py` is only the UI and all logic is in `modules/`.
4. **Honesty (25 s):** "OCR makes mistakes, my training data is synthetic so the accuracy is optimistic, and the summaries are extractive, not generative. The next step is training on real labelled documents."

## 5. Where AI, ML, NLP and OCR are used

| Area | Where in DocuMind | What it means here |
|---|---|---|
| **AI** (broad field) | The whole system | Software that performs tasks needing language/vision understanding: reading images, assigning topics, selecting key sentences, retrieving answers. |
| **ML** (learning from data) | `classifier.py` (Logistic Regression / Naive Bayes trained on labelled text); embeddings model in semantic search | The category rules are *learned* from examples, not hand-written. Evaluation uses a held-out test split. |
| **NLP** (processing language) | `text_preprocessor.py`, TF-IDF, TextRank in `summarizer.py`, search in `semantic_search.py`, keywords in `document_insights.py` | Tokenising, stop-word removal, sentence splitting, vectorising text, measuring similarity. |
| **OCR** (image → text) | `ocr_processor.py` (Tesseract), used by `pdf_processor.py` for scanned pages | Recognises characters in pictures so scanned documents become searchable text. |
| *Not ML* | PDF text-layer reading, sentence splitting, counting, keyword frequency | Deterministic rules and counting — labelling these "AI" would be overselling. |

## 6. Concepts in plain language

- **Stop-words:** very common words (*the, is, of*) that carry little topic information.
- **TF-IDF:** a word's weight = how often it appears in this document × how rare it is across all documents. Rare-but-repeated words (*attendance*) score high.
- **Cosine similarity:** compares the *direction* of two vectors; 1 = same direction (very similar), 0 = unrelated.
- **Logistic Regression:** learns a weight per word per category and turns the weighted sum into probabilities.
- **Naive Bayes:** uses word-count probabilities per category and assumes words are independent ("naive").
- **Extractive vs. abstractive summary:** extractive copies existing sentences; abstractive (generative AI) writes new sentences.
- **TextRank:** PageRank (Google's original algorithm) applied to sentences: a sentence is important if many important sentences are similar to it.
- **Embedding:** a list of numbers (here 384) representing meaning, so similar meanings sit close together.
- **Held-out test set:** data the model never saw in training, used to estimate how well it generalises.

## 7. Ten likely viva questions and answers

**1. Why did you use TF-IDF instead of simply counting words?**  
Raw counts reward common words like "the". TF-IDF multiplies a word's frequency in the document by how rare it is across all documents, so topic-specific words get more weight. I also use sublinear scaling (1 + log tf) so one word repeated 50 times doesn't dominate.

**2. Which algorithm did you choose for classification and why?**  
I train both Logistic Regression and Multinomial Naive Bayes on the training split, compare macro-F1 on the held-out split, and keep the better one; ties go to Logistic Regression because its probabilities are usually better behaved and it exposes learned word weights I use for explanations. On my synthetic data both scored the same, so Logistic Regression was selected.

**3. Your accuracy is very high. Is the model really that good?**  
No, and I don't claim that. The dataset is **synthetic**: I wrote sentences per category and combined them into documents, with different sentence pools for train and test so there is no direct leakage. Even so, the vocabulary is far cleaner than real documents, so the number is optimistic. The honest next step is training and cross-validating on real labelled documents. The metrics shown in the app come from `metrics.json`, which is produced by actual evaluation, not typed in.

**4. How does the app know whether a PDF is digital or scanned?**  
Per page, I read the text layer with PyMuPDF. If a page yields fewer than 25 characters, I treat it as an image, render it at 200 DPI and run Tesseract OCR on it. So a PDF with mixed pages is handled page by page.

**5. How does OCR work, and why can it be wrong?**  
Tesseract finds lines and words in the image, then recognises characters with a trained neural-network engine (an LSTM in Tesseract 4 and later). Errors come from blur, skew, low contrast, unusual fonts, handwriting and noise. I add light preprocessing (grayscale, contrast, upscaling small images) and display Tesseract's confidence as a warning sign, but it isn't a guarantee.

**6. Explain your summarisation method. Is it AI?**  
It's extractive TextRank: sentences become TF-IDF vectors, cosine similarity links them in a graph, and PageRank scores each sentence by how central it is. I return the top sentences in original order. It uses an algorithm, but it's not generative AI: it can't rephrase or combine ideas, and I state that in the app. For long documents I rank windows of 150 sentences so the matrix stays small and the summary covers the whole document.

**7. What is the difference between your semantic search and the TF-IDF fallback?**  
TF-IDF matches *words*, so "minimum presence" won't find "attendance". Semantic search uses Sentence-Transformer embeddings, which place similar meanings close together, so it can match different wording. Both rank passages by cosine similarity. The semantic option needs an extra install and a one-time model download; the fallback works offline with no extra packages.

**8. Why do you split the document into chunks for search?**  
Searching the whole document would only say "this document". Chunks of about 45 words with a one-sentence overlap let me return the exact passage that answers a question. I tested 70, 45 and 35 words on the demo document; 70 ranked an unrelated passage first for the attendance question, 45 fixed it. That is a small test, not a proven optimum.

**9. How do you explain a classification result to the user?**  
I show the probability for every category and the words that contributed most to the predicted class (for Logistic Regression: word weight × TF-IDF value). It's a simple explanation, not a full causal one. I also warn when confidence is below 50% and refuse to guess when there are fewer than five informative words.

**10. What are the limitations and how would you improve the project?**  
OCR errors; a small synthetic training set; broad categories that overlap (Business vs. Finance); extractive summaries; English only; no table or layout understanding. Improvements: train on a real labelled corpus with cross-validation, add stemming/lemmatisation, add Tesseract language packs, try a transformer classifier, use cross-encoder re-ranking for search, and add layout-aware extraction for tables.

## 8. Short answers to common follow-ups

- **"Why not use ChatGPT/an LLM API?"** The brief requires a real document-processing pipeline without paid keys, and local processing keeps documents private.
- **"Does it send my documents anywhere?"** No. The only network use is the optional one-time embedding-model download.
- **"Can you walk me through the code?"** Go in pipeline order: `pdf_processor.py` → `ocr_processor.py` → `text_preprocessor.py` → `classifier.py` → `summarizer.py` → `semantic_search.py` → `document_insights.py`, then `app.py` last. Read each file once before your viva so you can explain every function in your own words.
