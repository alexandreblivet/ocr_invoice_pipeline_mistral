"""
End-to-end smoke tests that hit the real Mistral API.

Run with:  uv run pytest tests/test_smoke.py -v -m smoke
Skip with: uv run pytest -m "not smoke"
"""
import os
from pathlib import Path

import pytest

from pipeline.extract import process_invoice
from pipeline.schemas import ProcessedInvoice

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"
API_KEY = os.getenv("MISTRAL_API_KEY", "")

pytestmark = pytest.mark.smoke


@pytest.fixture(scope="module")
def api_key():
    if not API_KEY:
        pytest.skip("MISTRAL_API_KEY not set")
    return API_KEY


def sample_pdfs():
    return sorted(SAMPLES_DIR.glob("*.pdf"))


@pytest.fixture(scope="module", params=[p.name for p in sample_pdfs()])
def sample_pdf(request):
    path = SAMPLES_DIR / request.param
    return path.name, path.read_bytes()


@pytest.mark.timeout(120)
def test_process_invoice_e2e(api_key, sample_pdf):
    filename, pdf_bytes = sample_pdf
    result = process_invoice(pdf_bytes, filename, api_key)

    assert isinstance(result, ProcessedInvoice)
    assert result.filename == filename
    assert result.extraction_method in ("annotation", "chat_parse")

    data = result.data
    assert data.vendor_name
    assert data.invoice_number is not None  # may be empty for template invoices
    assert data.invoice_date
    assert data.currency
    assert len(data.line_items) >= 1
    assert data.total_amount > 0
    assert data.subtotal > 0

    assert result.raw_markdown, "Expected OCR markdown output"
