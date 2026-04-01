from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from pipeline.rag import (
    QueryIntent,
    classify_query,
    build_sql_query,
    analytical_retrieval,
    semantic_retrieval,
    build_rag_context,
)
from pipeline.database import InvoiceDatabase
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


def _make_processed(filename="test.pdf", **overrides) -> ProcessedInvoice:
    data_dict = {**SAMPLE_INVOICE_DICT, **overrides}
    return ProcessedInvoice(
        filename=filename,
        data=InvoiceData(**data_dict),
        raw_markdown="# Invoice",
        extraction_method="annotation",
    )


def _mock_chat_response(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


@pytest.fixture
def db(tmp_path):
    return InvoiceDatabase(db_path=tmp_path / "test.db")


class TestClassifyQuery:
    @patch("pipeline.rag.Mistral")
    def test_analytical(self, MockMistral):
        client = MockMistral.return_value
        client.chat.complete.return_value = _mock_chat_response("analytical")

        result = classify_query("total spend by vendor", "fake-key")
        assert result == QueryIntent.ANALYTICAL

    @patch("pipeline.rag.Mistral")
    def test_semantic(self, MockMistral):
        client = MockMistral.return_value
        client.chat.complete.return_value = _mock_chat_response("semantic")

        result = classify_query("invoices about consulting", "fake-key")
        assert result == QueryIntent.SEMANTIC

    @patch("pipeline.rag.Mistral")
    def test_hybrid(self, MockMistral):
        client = MockMistral.return_value
        client.chat.complete.return_value = _mock_chat_response("hybrid")

        result = classify_query("cheapest vendor for supplies", "fake-key")
        assert result == QueryIntent.HYBRID

    @patch("pipeline.rag.Mistral")
    def test_defaults_to_hybrid_on_garbage(self, MockMistral):
        client = MockMistral.return_value
        client.chat.complete.return_value = _mock_chat_response("unknown_intent")

        result = classify_query("something weird", "fake-key")
        assert result == QueryIntent.HYBRID

    @patch("pipeline.rag.Mistral")
    def test_defaults_to_hybrid_on_error(self, MockMistral):
        client = MockMistral.return_value
        client.chat.complete.side_effect = Exception("API error")

        result = classify_query("anything", "fake-key")
        assert result == QueryIntent.HYBRID


class TestBuildSqlQuery:
    @patch("pipeline.rag.Mistral")
    def test_returns_sql(self, MockMistral):
        client = MockMistral.return_value
        client.chat.complete.return_value = _mock_chat_response(
            "SELECT vendor_name, SUM(total_amount) FROM invoices GROUP BY vendor_name"
        )

        sql = build_sql_query("total by vendor", "schema here", "fake-key")
        assert sql.startswith("SELECT")

    @patch("pipeline.rag.Mistral")
    def test_strips_code_fences(self, MockMistral):
        client = MockMistral.return_value
        client.chat.complete.return_value = _mock_chat_response(
            "```sql\nSELECT * FROM invoices\n```"
        )

        sql = build_sql_query("show all", "schema", "fake-key")
        assert "```" not in sql
        assert sql.startswith("SELECT")


class TestAnalyticalRetrieval:
    @patch("pipeline.rag.Mistral")
    def test_returns_results(self, MockMistral, db):
        db.save_invoice(_make_processed())
        client = MockMistral.return_value
        client.chat.complete.return_value = _mock_chat_response(
            "SELECT vendor_name, total_amount FROM invoices"
        )

        context, sql = analytical_retrieval("show invoices", db, "fake-key")
        assert "Acme Corp" in context
        assert "1650" in context

    @patch("pipeline.rag.Mistral")
    def test_fallback_on_bad_sql(self, MockMistral, db):
        db.save_invoice(_make_processed())
        client = MockMistral.return_value
        client.chat.complete.return_value = _mock_chat_response(
            "this is not sql"
        )

        context, sql = analytical_retrieval("anything", db, "fake-key")
        # Should fallback to all invoices
        assert "Acme Corp" in context


class TestSemanticRetrieval:
    @patch("pipeline.rag.embed_text")
    def test_with_vector_results(self, mock_embed, db):
        if not db._vec_available:
            pytest.skip("sqlite-vec not available")

        embedding = [1.0] + [0.0] * 1023
        db.save_invoice(_make_processed(), embedding=embedding)
        mock_embed.return_value = [0.9] + [0.1] + [0.0] * 1022

        context, n = semantic_retrieval("consulting work", db, "fake-key")
        assert n >= 1
        assert "Acme Corp" in context

    @patch("pipeline.rag.embed_text")
    def test_fallback_on_empty(self, mock_embed, db):
        mock_embed.return_value = [0.0] * 1024

        context, n = semantic_retrieval("anything", db, "fake-key")
        # Empty DB falls back to all invoices (which is also empty)
        assert n == 0


class TestBuildRagContext:
    def test_empty_db(self, db):
        context, intent, meta = build_rag_context("anything", db, "fake-key")
        assert context == ""
        assert meta == {}

    @patch("pipeline.rag.classify_query")
    @patch("pipeline.rag.analytical_retrieval")
    def test_routes_analytical(self, mock_retrieval, mock_classify, db):
        db.save_invoice(_make_processed())
        mock_classify.return_value = QueryIntent.ANALYTICAL
        mock_retrieval.return_value = ("SQL results here", "SELECT 1")

        context, intent, meta = build_rag_context("total spend", db, "fake-key")
        assert intent == QueryIntent.ANALYTICAL
        assert "SQL results here" in context
        assert meta["sql"] == "SELECT 1"

    @patch("pipeline.rag.classify_query")
    @patch("pipeline.rag.semantic_retrieval")
    def test_routes_semantic(self, mock_retrieval, mock_classify, db):
        db.save_invoice(_make_processed())
        mock_classify.return_value = QueryIntent.SEMANTIC
        mock_retrieval.return_value = ("Semantic results", 3)

        context, intent, meta = build_rag_context("find consulting", db, "fake-key")
        assert intent == QueryIntent.SEMANTIC
        assert "Semantic results" in context
        assert meta["num_results"] == 3
