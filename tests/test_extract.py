import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from pipeline.extract import (
    extract_with_annotation,
    extract_with_chat,
    process_invoice,
    EXTRACTION_PROMPT,
)
from pipeline.schemas import InvoiceData, ProcessedInvoice


SAMPLE_INVOICE_DICT = dict(
    vendor_name="Acme Corp",
    vendor_address="123 Main St",
    invoice_number="INV-001",
    invoice_date="2025-01-15",
    due_date="2025-02-15",
    currency="USD",
    line_items=[{"description": "Consulting", "quantity": 10, "unit_price": 150, "total": 1500}],
    subtotal=1500.0,
    tax_amount=150.0,
    total_amount=1650.0,
    payment_terms="Net 30",
)

SAMPLE_MARKDOWN = "# Invoice INV-001\nVendor: Acme Corp\nTotal: $1,650.00"


def _make_ocr_response(annotation, markdown=SAMPLE_MARKDOWN):
    """Build a fake OCR response object."""
    page = SimpleNamespace(markdown=markdown)
    return SimpleNamespace(
        pages=[page],
        document_annotation=annotation,
    )


def _make_chat_response(invoice_data: InvoiceData):
    """Build a fake chat.parse response."""
    choice = SimpleNamespace(message=SimpleNamespace(parsed=invoice_data))
    return SimpleNamespace(choices=[choice])


# --- extract_with_annotation ---

class TestExtractWithAnnotation:
    @patch("pipeline.extract._upload_and_get_url", return_value="https://fake-url")
    @patch("pipeline.extract.Mistral")
    def test_annotation_returns_json_string(self, MockMistral, mock_upload):
        annotation_json = json.dumps(SAMPLE_INVOICE_DICT)
        ocr_response = _make_ocr_response(annotation=annotation_json)

        client = MockMistral.return_value
        client.ocr.process.return_value = ocr_response

        data, md = extract_with_annotation(b"fake-pdf", "fake-key")
        assert isinstance(data, InvoiceData)
        assert data.vendor_name == "Acme Corp"
        assert data.total_amount == 1650.0
        assert md == SAMPLE_MARKDOWN

    @patch("pipeline.extract._upload_and_get_url", return_value="https://fake-url")
    @patch("pipeline.extract.Mistral")
    def test_annotation_returns_dict(self, MockMistral, mock_upload):
        ocr_response = _make_ocr_response(annotation=SAMPLE_INVOICE_DICT)
        client = MockMistral.return_value
        client.ocr.process.return_value = ocr_response

        data, md = extract_with_annotation(b"fake-pdf", "fake-key")
        assert data.invoice_number == "INV-001"

    @patch("pipeline.extract._upload_and_get_url", return_value="https://fake-url")
    @patch("pipeline.extract.Mistral")
    def test_annotation_none_raises(self, MockMistral, mock_upload):
        ocr_response = _make_ocr_response(annotation=None)
        client = MockMistral.return_value
        client.ocr.process.return_value = ocr_response

        with pytest.raises(ValueError, match="No document annotation"):
            extract_with_annotation(b"fake-pdf", "fake-key")


# --- extract_with_chat ---

class TestExtractWithChat:
    @patch("pipeline.extract.Mistral")
    def test_chat_parse_returns_invoice(self, MockMistral):
        expected = InvoiceData(**SAMPLE_INVOICE_DICT)
        client = MockMistral.return_value
        client.chat.parse.return_value = _make_chat_response(expected)

        result = extract_with_chat(SAMPLE_MARKDOWN, "fake-key")
        assert result == expected

    @patch("pipeline.extract.Mistral")
    def test_chat_parse_uses_correct_params(self, MockMistral):
        expected = InvoiceData(**SAMPLE_INVOICE_DICT)
        client = MockMistral.return_value
        client.chat.parse.return_value = _make_chat_response(expected)

        extract_with_chat(SAMPLE_MARKDOWN, "fake-key")

        call_kwargs = client.chat.parse.call_args.kwargs
        assert call_kwargs["model"] == "mistral-small-latest"
        assert call_kwargs["temperature"] == 0
        assert call_kwargs["response_format"] is InvoiceData
        messages = call_kwargs["messages"]
        assert messages[0]["content"] == EXTRACTION_PROMPT
        assert messages[1]["content"] == SAMPLE_MARKDOWN


# --- process_invoice (integration of both tiers) ---

class TestProcessInvoice:
    @patch("pipeline.extract.extract_with_annotation")
    def test_tier1_success(self, mock_annotation):
        inv_data = InvoiceData(**SAMPLE_INVOICE_DICT)
        mock_annotation.return_value = (inv_data, SAMPLE_MARKDOWN)

        result = process_invoice(b"fake-pdf", "test.pdf", "fake-key")
        assert isinstance(result, ProcessedInvoice)
        assert result.extraction_method == "annotation"
        assert result.data.vendor_name == "Acme Corp"

    @patch("pipeline.extract.extract_with_chat")
    @patch("pipeline.extract.get_markdown", return_value=SAMPLE_MARKDOWN)
    @patch("pipeline.extract.run_ocr")
    @patch("pipeline.extract.extract_with_annotation", side_effect=Exception("tier1 failed"))
    def test_fallback_to_tier2(self, mock_ann, mock_ocr, mock_md, mock_chat):
        inv_data = InvoiceData(**SAMPLE_INVOICE_DICT)
        mock_chat.return_value = inv_data

        result = process_invoice(b"fake-pdf", "test.pdf", "fake-key")
        assert result.extraction_method == "chat_parse"
        assert result.data == inv_data

    @patch("pipeline.extract.run_ocr", side_effect=Exception("OCR failed"))
    @patch("pipeline.extract.extract_with_annotation", side_effect=Exception("tier1 failed"))
    def test_both_tiers_fail_raises_runtime_error(self, mock_ann, mock_ocr):
        with pytest.raises(RuntimeError, match="Failed to extract invoice data from test.pdf"):
            process_invoice(b"fake-pdf", "test.pdf", "fake-key")
