from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from pipeline.embeddings import embed_text, embed_texts, build_invoice_chunk
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


def _make_embed_response(vectors: list[list[float]]):
    """Build a mock embeddings response."""
    data = [
        SimpleNamespace(embedding=vec, index=i)
        for i, vec in enumerate(vectors)
    ]
    return SimpleNamespace(data=data)


class TestEmbedText:
    @patch("pipeline.embeddings.Mistral")
    def test_returns_vector(self, MockMistral):
        fake_vector = [0.1] * 1024
        client = MockMistral.return_value
        client.embeddings.create.return_value = _make_embed_response([fake_vector])

        result = embed_text("hello", "fake-key")

        assert result == fake_vector
        assert len(result) == 1024
        client.embeddings.create.assert_called_once()

    @patch("pipeline.embeddings.Mistral")
    def test_passes_correct_model(self, MockMistral):
        client = MockMistral.return_value
        client.embeddings.create.return_value = _make_embed_response([[0.0] * 1024])

        embed_text("test", "fake-key")

        call_kwargs = client.embeddings.create.call_args
        assert call_kwargs.kwargs["model"] == "mistral-embed"
        assert call_kwargs.kwargs["inputs"] == ["test"]


class TestEmbedTexts:
    @patch("pipeline.embeddings.Mistral")
    def test_batch_returns_ordered(self, MockMistral):
        vecs = [[float(i)] * 1024 for i in range(3)]
        # Return in shuffled order to test sorting
        data = [
            SimpleNamespace(embedding=vecs[2], index=2),
            SimpleNamespace(embedding=vecs[0], index=0),
            SimpleNamespace(embedding=vecs[1], index=1),
        ]
        client = MockMistral.return_value
        client.embeddings.create.return_value = SimpleNamespace(data=data)

        result = embed_texts(["a", "b", "c"], "fake-key")

        assert len(result) == 3
        assert result[0] == vecs[0]
        assert result[1] == vecs[1]
        assert result[2] == vecs[2]


class TestBuildInvoiceChunk:
    def test_contains_key_fields(self):
        invoice = ProcessedInvoice(
            filename="test.pdf",
            data=InvoiceData(**SAMPLE_INVOICE_DICT),
            raw_markdown="Some OCR text here",
            extraction_method="annotation",
        )
        chunk = build_invoice_chunk(invoice, "Some OCR text here")

        assert "INV-001" in chunk
        assert "Acme Corp" in chunk
        assert "2025-01-15" in chunk
        assert "Consulting" in chunk
        assert "Net 30" in chunk
        assert "Some OCR text here" in chunk

    def test_truncates_long_markdown(self):
        long_markdown = "x" * 5000
        invoice = ProcessedInvoice(
            filename="test.pdf",
            data=InvoiceData(**SAMPLE_INVOICE_DICT),
            raw_markdown=long_markdown,
        )
        chunk = build_invoice_chunk(invoice, long_markdown)

        # Should truncate to 1500 chars of markdown
        assert len(chunk) < 5000

    def test_handles_no_payment_terms(self):
        data = {**SAMPLE_INVOICE_DICT, "payment_terms": None}
        invoice = ProcessedInvoice(
            filename="test.pdf",
            data=InvoiceData(**data),
            raw_markdown="text",
        )
        chunk = build_invoice_chunk(invoice, "text")
        assert "not specified" in chunk
