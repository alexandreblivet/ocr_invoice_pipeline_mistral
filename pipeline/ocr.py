from mistralai.client import Mistral
from mistralai.client.models import DocumentURLChunk


def _upload_and_get_url(client: Mistral, pdf_bytes: bytes, filename: str = "document.pdf") -> str:
    """Upload a file to Mistral and return a signed URL."""
    uploaded = client.files.upload(
        file={"file_name": filename, "content": pdf_bytes, "content_type": "application/pdf"},
        purpose="ocr",
    )
    signed = client.files.get_signed_url(file_id=uploaded.id, expiry=1)
    return signed.url


def run_ocr(pdf_bytes: bytes, api_key: str, filename: str = "document.pdf"):
    """Run Mistral OCR on a PDF and return the raw OCRResponse."""
    client = Mistral(api_key=api_key)
    url = _upload_and_get_url(client, pdf_bytes, filename)
    return client.ocr.process(
        model="mistral-ocr-latest",
        document=DocumentURLChunk(document_url=url),
        table_format="html",
        include_image_base64=True,
    )


def get_markdown(ocr_response) -> str:
    """Extract full markdown text from an OCR response."""
    parts = []
    for i, page in enumerate(ocr_response.pages):
        if i > 0:
            parts.append(f"\n\n--- Page {i + 1} ---\n\n")
        parts.append(page.markdown)
    return "".join(parts)
