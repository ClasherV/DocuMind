"""
OCR (Optical Character Recognition) with Tesseract.

OCR turns a picture of text into real text. Tesseract is a separate program
(not a Python package) that must be installed on your computer; `pytesseract`
is just the small Python wrapper that calls it. Everything runs locally -
no image is ever uploaded anywhere.
"""
import io
import os
import shutil
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, ImageOps, UnidentifiedImageError

from modules.extraction_result import ExtractionResult, PageResult

try:
    import pytesseract
    _PYTESSERACT_IMPORTED = True
except ImportError:  # the app still starts; OCR features just report a clear error
    pytesseract = None
    _PYTESSERACT_IMPORTED = False

# Places where the Windows installer normally puts tesseract.exe
_WINDOWS_CANDIDATES = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    str(Path.home() / "AppData" / "Local" / "Programs" / "Tesseract-OCR" / "tesseract.exe"),
]

MAX_IMAGE_PIXELS = 40_000_000   # refuse absurdly large images (memory protection)


def find_tesseract() -> Optional[str]:
    """Locate tesseract.exe. Order: TESSERACT_CMD env var -> PATH -> default Windows folders."""
    env_path = os.environ.get("TESSERACT_CMD")
    if env_path and Path(env_path).is_file():
        return env_path
    on_path = shutil.which("tesseract")
    if on_path:
        return on_path
    for candidate in _WINDOWS_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    return None


def configure_tesseract() -> Tuple[bool, str]:
    """Point pytesseract at the executable. Returns (available, human-readable message)."""
    if not _PYTESSERACT_IMPORTED:
        return False, "The 'pytesseract' Python package is not installed (pip install pytesseract)."
    path = find_tesseract()
    if path is None:
        return False, (
            "Tesseract OCR was not found. Install it (see README) or set the TESSERACT_CMD "
            "environment variable to the full path of tesseract.exe."
        )
    pytesseract.pytesseract.tesseract_cmd = path
    try:
        version = pytesseract.get_tesseract_version()
        return True, f"Tesseract {version} found at {path}"
    except Exception as exc:  # noqa: BLE001 - we want to report any launch problem
        return False, f"Tesseract was found at {path} but could not be started: {exc}"


def _prepare_image(image: Image.Image) -> Image.Image:
    """Light cleaning that usually helps OCR: fix rotation, grayscale, upscale small images."""
    image = ImageOps.exif_transpose(image)           # honour phone-camera rotation
    if image.mode not in ("L", "RGB"):
        image = image.convert("RGB")
    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray)               # stretch contrast for faded scans
    if max(gray.size) < 1200:                        # Tesseract likes ~300 DPI text
        scale = 1200 / max(gray.size)
        gray = gray.resize((int(gray.width * scale), int(gray.height * scale)), Image.LANCZOS)
    return gray


def ocr_image(image: Image.Image, lang: str = "eng") -> Tuple[str, Optional[float]]:
    """
    Run OCR on a PIL image.
    Returns (text, mean_word_confidence_0_to_100). Confidence is Tesseract's own
    estimate - useful as a warning sign, but not a guarantee of correctness.
    """
    available, message = configure_tesseract()
    if not available:
        raise RuntimeError(message)

    prepared = _prepare_image(image)
    text = pytesseract.image_to_string(prepared, lang=lang)

    confidence = None
    try:
        data = pytesseract.image_to_data(prepared, lang=lang, output_type=pytesseract.Output.DICT)
        scores = [
            float(c) for c, w in zip(data["conf"], data["text"])
            if str(w).strip() and float(c) >= 0
        ]
        if scores:
            confidence = sum(scores) / len(scores)
    except Exception:  # noqa: BLE001 - confidence is a bonus; never fail because of it
        confidence = None
    return text, confidence


def extract_text_from_image(file_bytes: bytes) -> ExtractionResult:
    """Validate an uploaded image (JPG/PNG) and extract its text with OCR."""
    if not file_bytes:
        return ExtractionResult(ok=False, source_type="image", error="The uploaded image file is empty.")

    try:
        probe = Image.open(io.BytesIO(file_bytes))
        probe.verify()                                # detects truncated / corrupt files
        image = Image.open(io.BytesIO(file_bytes))    # re-open: verify() invalidates the object
        image.load()
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        return ExtractionResult(
            ok=False, source_type="image",
            error=f"This file could not be read as an image (it may be corrupted): {exc}",
        )

    if image.width * image.height > MAX_IMAGE_PIXELS:
        return ExtractionResult(ok=False, source_type="image",
                                error="The image is too large to process safely. Please downscale it.")

    try:
        text, confidence = ocr_image(image)
    except RuntimeError as exc:
        return ExtractionResult(ok=False, source_type="image", error=str(exc))
    except Exception as exc:  # noqa: BLE001
        return ExtractionResult(ok=False, source_type="image", error=f"OCR failed unexpectedly: {exc}")

    result = ExtractionResult(ok=True, source_type="image", text=text.strip(),
                              pages=[PageResult(1, "ocr", text.strip(), confidence)])
    if not result.text:
        result.ok = False
        result.error = "OCR finished but found no readable text. Try a sharper, higher-contrast image."
    elif confidence is not None and confidence < 60:
        result.warnings.append(
            f"Low OCR confidence ({confidence:.0f}/100). The text probably contains mistakes - "
            "check it before trusting the results."
        )
    return result
