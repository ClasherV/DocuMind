"""Shared data containers returned by the PDF and OCR processors."""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PageResult:
    """Text extracted from one PDF page (or the single 'page' of an image)."""
    page_number: int
    method: str                              # "digital" (PDF text layer) or "ocr" (Tesseract)
    text: str
    ocr_confidence: Optional[float] = None   # 0-100, only for OCR pages


@dataclass
class ExtractionResult:
    """Everything the rest of the app needs to know about one extraction."""
    ok: bool
    text: str = ""
    source_type: str = ""                    # "pdf" or "image"
    pages: List[PageResult] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def methods_used(self) -> str:
        used = sorted({p.method for p in self.pages})
        return " + ".join(used) if used else "none"

    @property
    def mean_ocr_confidence(self) -> Optional[float]:
        values = [p.ocr_confidence for p in self.pages if p.ocr_confidence is not None]
        return sum(values) / len(values) if values else None
