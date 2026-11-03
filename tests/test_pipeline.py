"""
DocuMind tests.  Run from the project folder with:

    python -m unittest discover -s tests -v

Tests that need an optional component are SKIPPED (not faked) when it is missing:
  * PDF tests   need PyMuPDF
  * OCR tests   need Tesseract
  * semantic    needs sentence-transformers (+ model download)

All sample files used here are DEMO documents from data/sample_documents/.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modules import classifier, ocr_processor, pdf_processor, summarizer          # noqa: E402
from modules import document_insights as insights                                   # noqa: E402
from modules.semantic_search import DocumentSearchEngine, is_semantic_available     # noqa: E402
from modules.text_preprocessor import make_chunks, split_sentences                  # noqa: E402

SAMPLES = ROOT / "data" / "sample_documents"
INVALID = SAMPLES / "invalid"

HAS_PYMUPDF = pdf_processor._FITZ_IMPORTED
HAS_TESSERACT = ocr_processor.configure_tesseract()[0]


def read(path: Path) -> bytes:
    return path.read_bytes()


def sample_text(name: str) -> str:
    """Extract text from a demo file, skipping the test if the needed tool is unavailable."""
    path = SAMPLES / name
    if path.suffix == ".pdf":
        if not HAS_PYMUPDF:
            raise unittest.SkipTest("PyMuPDF not installed")
        result = pdf_processor.extract_text_from_pdf(read(path))
    else:
        if not HAS_TESSERACT:
            raise unittest.SkipTest("Tesseract not installed")
        result = ocr_processor.extract_text_from_image(read(path))
    assert result.ok, result.error
    return result.text


class TestExtraction(unittest.TestCase):
    @unittest.skipUnless(HAS_PYMUPDF, "PyMuPDF not installed")
    def test_digital_pdf_uses_text_layer(self):
        result = pdf_processor.extract_text_from_pdf(read(SAMPLES / "demo_academic_regulations.pdf"))
        self.assertTrue(result.ok)
        self.assertEqual(result.methods_used, "digital")
        self.assertIn("attendance", result.text.lower())

    @unittest.skipUnless(HAS_PYMUPDF and HAS_TESSERACT, "needs PyMuPDF and Tesseract")
    def test_scanned_pdf_falls_back_to_ocr(self):
        result = pdf_processor.extract_text_from_pdf(read(SAMPLES / "demo_scanned_business_memo.pdf"))
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.methods_used, "ocr")
        self.assertIn("marketing", result.text.lower())

    @unittest.skipUnless(HAS_TESSERACT, "Tesseract not installed")
    def test_image_ocr(self):
        result = ocr_processor.extract_text_from_image(read(SAMPLES / "demo_legal_agreement.png"))
        self.assertTrue(result.ok, result.error)
        self.assertIn("agreement", result.text.lower())
        self.assertIsNotNone(result.mean_ocr_confidence)


class TestInvalidFiles(unittest.TestCase):
    def test_empty_files(self):
        self.assertFalse(pdf_processor.extract_text_from_pdf(b"").ok)
        self.assertFalse(ocr_processor.extract_text_from_image(b"").ok)

    def test_corrupted_pdf(self):
        result = pdf_processor.extract_text_from_pdf(read(INVALID / "corrupted.pdf"))
        self.assertFalse(result.ok)
        self.assertTrue(result.error)

    def test_fake_pdf_without_header(self):
        self.assertFalse(pdf_processor.extract_text_from_pdf(b"just some text").ok)

    def test_corrupted_image(self):
        result = ocr_processor.extract_text_from_image(read(INVALID / "not_an_image.png"))
        self.assertFalse(result.ok)
        self.assertIn("could not be read", result.error)

    @unittest.skipUnless(HAS_TESSERACT, "Tesseract not installed")
    def test_blank_image_reports_no_text(self):
        result = ocr_processor.extract_text_from_image(read(INVALID / "blank_image.png"))
        self.assertFalse(result.ok)


class TestPreprocessing(unittest.TestCase):
    def test_sentence_splitter_protects_abbreviations_and_decimals(self):
        text = "Dr. Rao scored 8.5 points. She passed the exam. Was it hard? Yes!"
        self.assertEqual(len(split_sentences(text)), 4)

    def test_chunks_cover_text(self):
        text = " ".join(f"Sentence number {i} talks about topic {i}." for i in range(40))
        chunks = make_chunks(text, max_words=30)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c.split()) <= 45 for c in chunks))


class TestClassification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = classifier.load_or_train()

    def test_each_category_on_typical_text(self):
        cases = {
            "Education": "Students must keep minimum attendance to appear for the semester examination and the professor sets the syllabus.",
            "Finance": "The bank raised the interest rate on loans, and the balance sheet shows higher liabilities and lower profit.",
            "Technology": "The server exposes an API, stores records in a database and uses encryption to protect network traffic.",
            "Legal": "The parties agree that the tenant shall indemnify the landlord and any dispute is under the jurisdiction of the court.",
            "Business": "The marketing team will launch a customer campaign and the sales director expects strong demand for the product.",
        }
        for expected, text in cases.items():
            with self.subTest(expected=expected):
                self.assertEqual(classifier.predict(self.bundle, text)["label"], expected)

    def test_probabilities_sum_to_one(self):
        result = classifier.predict(self.bundle, "The bank raised interest rates on home loans this quarter.")
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0, places=5)
        self.assertGreater(len(result["top_terms"]), 0)

    def test_too_little_text_is_not_guessed(self):
        result = classifier.predict(self.bundle, "hello there")
        self.assertTrue(result["insufficient_text"])

    def test_demo_documents(self):
        expected = {"demo_academic_regulations.pdf": "Education", "demo_finance_report.pdf": "Finance",
                    "demo_technology_note.pdf": "Technology", "demo_legal_agreement.png": "Legal",
                    "demo_scanned_business_memo.pdf": "Business"}
        for name, label in expected.items():
            with self.subTest(file=name):
                text = sample_text(name)
                self.assertEqual(classifier.predict(self.bundle, text)["label"], label)

    def test_metrics_file_matches_model(self):
        metrics = classifier.load_metrics()
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics["selected_algorithm"], self.bundle["algorithm"])


class TestSummarization(unittest.TestCase):
    TEXT = " ".join(
        [f"The attendance rule number {i} explains how students are assessed during the semester." for i in range(10)]
        + [f"The hostel office handles room allocation request number {i} for residents." for i in range(10)]
        + [f"The library offers digital journals and reference books to every learner, item {i}." for i in range(10)])

    def test_summary_is_shorter_and_uses_original_sentences(self):
        result = summarizer.summarize(self.TEXT, ratio=0.2)
        original = set(split_sentences(self.TEXT))
        self.assertLess(result["summary_words"], result["original_words"])
        self.assertTrue(all(s in original for s in result["sentences"]))

    def test_length_control(self):
        short = summarizer.summarize(self.TEXT, ratio=0.1)
        longer = summarizer.summarize(self.TEXT, ratio=0.5)
        self.assertLess(len(short["sentences"]), len(longer["sentences"]))
        self.assertLessEqual(len(summarizer.summarize(self.TEXT, ratio=0.5, max_sentences=2)["sentences"]), 2)

    def test_long_document_is_handled(self):
        result = summarizer.summarize((self.TEXT + " ") * 40, ratio=0.05)
        self.assertGreater(result["total_sentences"], 1000)
        self.assertGreater(len(result["sentences"]), 0)

    def test_empty_text(self):
        self.assertEqual(summarizer.summarize("")["summary"], "")


class TestSearch(unittest.TestCase):
    TEXT = ("Attendance Requirement\n\nEvery student must maintain a minimum attendance of 75 percent to appear "
            "for the examination.\n\nGrading System\n\nGrades are awarded on a ten point scale and the CGPA is "
            "computed using course credits.\n\nLibrary\n\nThe library is open from 8 am to 10 pm on working days.")

    def test_tfidf_finds_the_right_passage(self):
        engine = DocumentSearchEngine(self.TEXT, backend="tfidf", max_words=20)
        results = engine.search("What is the attendance requirement?", top_k=1)
        self.assertTrue(results)
        self.assertIn("attendance", results[0]["text"].lower())

    def test_tfidf_returns_nothing_for_unrelated_query(self):
        engine = DocumentSearchEngine(self.TEXT, backend="tfidf")
        self.assertEqual(engine.search("pizza recipe with mushrooms"), [])

    @unittest.skipUnless(is_semantic_available(), "sentence-transformers not installed")
    def test_semantic_matches_meaning(self):
        try:
            engine = DocumentSearchEngine(self.TEXT, backend="semantic", max_words=20)
        except Exception as exc:  # e.g. offline, model cannot be downloaded
            self.skipTest(f"embedding model unavailable: {exc}")
        results = engine.search("minimum presence needed in class", top_k=1)
        self.assertIn("attendance", results[0]["text"].lower())


class TestInsights(unittest.TestCase):
    def test_counts(self):
        text = "First paragraph here. It has two sentences.\n\nSecond paragraph."
        stats = insights.compute_insights(text)
        self.assertEqual(stats["words"], 9)
        self.assertEqual(stats["sentences"], 3)
        self.assertEqual(stats["paragraphs"], 2)
        self.assertEqual(stats["characters"], len(text))

    def test_keywords_ignore_stop_words(self):
        words = [w for w, _ in insights.top_keywords("The cat and the cat and the dog chase the cat.")]
        self.assertEqual(words[0], "cat")
        self.assertNotIn("the", words)


if __name__ == "__main__":
    unittest.main(verbosity=2)
