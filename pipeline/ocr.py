import logging
import time

from mistralai.client import Mistral
from mistralai.client.models import DocumentURLChunk

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BACKOFF = (1, 3, 8)  # seconds between retries
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _is_retryable(exc: Exception) -> bool:
    """Check if an exception is worth retrying (rate limit or server error)."""
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status and int(status) in RETRYABLE_STATUS_CODES:
        return True
    msg = str(exc).lower()
    return any(term in msg for term in ("rate limit", "timeout", "timed out", "connection"))


def _retry(func, *args, context: str = "", **kwargs):
    """Call func with retries on transient failures."""
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES - 1 and _is_retryable(exc):
                wait = RETRY_BACKOFF[attempt]
                logger.warning(
                    "%s: attempt %d/%d failed (%s), retrying in %ds",
                    context, attempt + 1, MAX_RETRIES, exc, wait,
                )
                time.sleep(wait)
            else:
                break
    raise last_exc


def _upload_and_get_url(client: Mistral, pdf_bytes: bytes, filename: str = "document.pdf") -> str:
    """Upload a file to Mistral and return a signed URL."""
    logger.info("Uploading %s (%d bytes)", filename, len(pdf_bytes))
    uploaded = _retry(
        client.files.upload,
        file={"file_name": filename, "content": pdf_bytes, "content_type": "application/pdf"},
        purpose="ocr",
        context=f"upload({filename})",
    )
    signed = client.files.get_signed_url(file_id=uploaded.id, expiry=1)
    return signed.url


def run_ocr(pdf_bytes: bytes, api_key: str, filename: str = "document.pdf"):
    """Run Mistral OCR on a PDF and return the raw OCRResponse."""
    client = Mistral(api_key=api_key)
    url = _upload_and_get_url(client, pdf_bytes, filename)
    logger.info("Running OCR on %s", filename)
    return _retry(
        client.ocr.process,
        model="mistral-ocr-latest",
        document=DocumentURLChunk(document_url=url),
        table_format="html",
        include_image_base64=True,
        context=f"ocr({filename})",
    )


def get_markdown(ocr_response) -> str:
    """Extract full markdown text from an OCR response."""
    parts = []
    for i, page in enumerate(ocr_response.pages):
        if i > 0:
            parts.append(f"\n\n--- Page {i + 1} ---\n\n")
        parts.append(page.markdown)
    return "".join(parts)
