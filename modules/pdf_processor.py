"""
PDF text extraction with PyMuPDF (imported as `fitz`).

Two kinds of PDF exist:
  1. Digital PDFs  - contain a real text layer, so we can read the text directly (fast, exact).
  2. Scanned PDFs  - each page is just a picture. There is no text to read, so we render the
                     page to an image and run OCR on it (slower, may contain mistakes).

This module checks every page separately and only uses OCR where the text layer is missing.
"""
import io
import re
from typing import List

from PIL import Image

from modules.extraction_result import ExtractionResult, PageResult
from modules import ocr_processor

try:
    import fitz  # PyMuPDF
    _FITZ_IMPORTED = True
except ImportError:
    fitz = None
    _FITZ_IMPORTED = False

MAX_PAGES = 60              # safety limit so one huge PDF cannot freeze the app
MIN_CHARS_FOR_TEXT_LAYER = 25   # fewer characters than this on a page -> treat page as scanned
OCR_DPI = 200               # resolution used when rendering a page for OCR


def _clean_block(text: str) -> str:
    """Join a PDF text block's lines into one paragraph and repair hyphenated line breaks."""
    text = text.replace("\u00ad", "")                      # soft hyphens
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)            # "exami-\nnation" -> "examination"
    text = re.sub(r"\s*\n\s*", " ", text)                   # single line breaks -> spaces
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def _digital_page_text(page) -> str:
    """Read the text layer of one page, one paragraph per PDF text block."""
    blocks = page.get_text("blocks", sort=True)             # reading order (top-to-bottom)
    paragraphs: List[str] = []
    for block in blocks:
        if len(block) >= 7 and block[6] != 0:               # block_type 0 = text, 1 = image
            continue
        cleaned = _clean_block(block[4])
        if cleaned:
            paragraphs.append(cleaned)
    return "\n\n".join(paragraphs)


def _ocr_page(page) -> PageResult:
    """Render a page to an image and OCR it."""
    pixmap = page.get_pixmap(dpi=OCR_DPI)
    image = Image.open(io.BytesIO(pixmap.tobytes("png")))
    text, confidence = ocr_processor.ocr_image(image)
    return PageResult(page.number + 1, "ocr", text.strip(), confidence)


def extract_text_from_pdf(file_bytes: bytes) -> ExtractionResult:
    """Extract text from PDF bytes. Never raises: problems are reported in the result."""
    if not _FITZ_IMPORTED:
        return ExtractionResult(ok=False, source_type="pdf",
                                error="PyMuPDF is not installed. Run: pip install PyMuPDF")
    if not file_bytes:
        return ExtractionResult(ok=False, source_type="pdf", error="The uploaded PDF file is empty.")
    if not file_bytes.lstrip()[:5] == b"%PDF-":
        return ExtractionResult(ok=False, source_type="pdf",
                                error="This file does not look like a real PDF (missing %PDF header).")

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:  # noqa: BLE001 - PyMuPDF raises several exception types
        return ExtractionResult(ok=False, source_type="pdf",
                                error=f"The PDF could not be opened (it may be corrupted): {exc}")

    result = ExtractionResult(ok=True, source_type="pdf")
    try:
        if doc.needs_pass:
            return ExtractionResult(ok=False, source_type="pdf",
                                    error="This PDF is password-protected. Remove the password and try again.")
        if doc.page_count == 0:
            return ExtractionResult(ok=False, source_type="pdf", error="The PDF contains no pages.")

        total_pages = doc.page_count
        if total_pages > MAX_PAGES:
            result.warnings.append(f"Only the first {MAX_PAGES} of {total_pages} pages were processed.")

        ocr_unavailable_noted = False
        for index in range(min(total_pages, MAX_PAGES)):
            page = doc[index]
            try:
                text = _digital_page_text(page)
                if len(text.strip()) >= MIN_CHARS_FOR_TEXT_LAYER:
                    result.pages.append(PageResult(index + 1, "digital", text))
                    continue
                # No usable text layer -> this page is probably a scan; try OCR.
                try:
                    result.pages.append(_ocr_page(page))
                except RuntimeError as exc:                 # Tesseract missing
                    if not ocr_unavailable_noted:
                        result.warnings.append(f"Scanned page(s) skipped: {exc}")
                        ocr_unavailable_noted = True
                    result.pages.append(PageResult(index + 1, "digital", text))
            except Exception as exc:  # noqa: BLE001 - one bad page must not kill the document
                result.warnings.append(f"Page {index + 1} could not be read: {exc}")
    finally:
        doc.close()

    result.text = "\n\n".join(p.text for p in result.pages if p.text.strip()).strip()
    if not result.text:
        result.ok = False
        result.error = ("No text could be extracted. The PDF may be blank, or it is scanned and "
                        "Tesseract OCR is unavailable.")
        return result

    ocr_scores = [p.ocr_confidence for p in result.pages if p.method == "ocr" and p.ocr_confidence is not None]
    if ocr_scores and (sum(ocr_scores) / len(ocr_scores)) < 60:
        result.warnings.append("Low OCR confidence on scanned pages - expect recognition mistakes.")
    return result
