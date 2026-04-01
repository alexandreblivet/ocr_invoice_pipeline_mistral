import pytest

from pipeline.database import InvoiceDatabase, _serialize_f32
from pipeline.schemas import InvoiceData, LineItem, ProcessedInvoice


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


@pytest.fixture
def db(tmp_path):
    return InvoiceDatabase(db_path=tmp_path / "test.db")


class TestSaveAndRetrieve:
    def test_save_and_get_all(self, db):
        inv = _make_processed()
        invoice_id = db.save_invoice(inv, chunk_text="test chunk")

        assert invoice_id is not None
        all_inv = db.get_all_invoices()
        assert len(all_inv) == 1
        assert all_inv[0]["vendor_name"] == "Acme Corp"
        assert all_inv[0]["invoice_number"] == "INV-001"
        assert all_inv[0]["line_items_count"] == 1

    def test_save_with_embedding(self, db):
        inv = _make_processed()
        embedding = [0.1] * 1024
        invoice_id = db.save_invoice(inv, embedding=embedding)

        assert invoice_id is not None

    def test_get_by_filename(self, db):
        db.save_invoice(_make_processed("a.pdf"))
        db.save_invoice(_make_processed("b.pdf", vendor_name="Other Corp"))

        result = db.get_invoice_by_filename("b.pdf")
        assert result is not None
        assert result["vendor_name"] == "Other Corp"

    def test_get_by_filename_not_found(self, db):
        assert db.get_invoice_by_filename("nope.pdf") is None

    def test_resave_same_filename(self, db):
        db.save_invoice(_make_processed("dup.pdf", total_amount=100.0))
        db.save_invoice(_make_processed("dup.pdf", total_amount=200.0))

        all_inv = db.get_all_invoices()
        assert len(all_inv) == 1
        assert all_inv[0]["total_amount"] == 200.0


class TestLineItems:
    def test_get_line_items(self, db):
        inv = _make_processed()
        invoice_id = db.save_invoice(inv)

        items = db.get_line_items(invoice_id)
        assert len(items) == 1
        assert items[0]["description"] == "Consulting"
        assert items[0]["quantity"] == 10.0
        assert items[0]["unit_price"] == 150.0
        assert items[0]["total"] == 1500.0

    def test_multiple_line_items(self, db):
        data = {
            **SAMPLE_INVOICE_DICT,
            "line_items": [
                {"description": "Item A", "quantity": 1, "unit_price": 100, "total": 100},
                {"description": "Item B", "quantity": 2, "unit_price": 50, "total": 100},
            ],
        }
        inv = ProcessedInvoice(
            filename="multi.pdf",
            data=InvoiceData(**data),
            raw_markdown="",
            extraction_method="annotation",
        )
        invoice_id = db.save_invoice(inv)
        items = db.get_line_items(invoice_id)
        assert len(items) == 2


class TestDelete:
    def test_delete_invoice(self, db):
        invoice_id = db.save_invoice(_make_processed())
        db.delete_invoice(invoice_id)

        assert db.is_empty
        assert db.get_line_items(invoice_id) == []

    def test_delete_with_embedding(self, db):
        embedding = [0.5] * 1024
        invoice_id = db.save_invoice(_make_processed(), embedding=embedding)
        db.delete_invoice(invoice_id)
        assert db.is_empty


class TestVectorSearch:
    def test_search_similar(self, db):
        if not db._vec_available:
            pytest.skip("sqlite-vec not available")

        # Insert two invoices with distinct embeddings
        vec_a = [1.0] + [0.0] * 1023
        vec_b = [0.0] + [1.0] + [0.0] * 1022
        db.save_invoice(_make_processed("a.pdf"), embedding=vec_a)
        db.save_invoice(_make_processed("b.pdf", vendor_name="Other"), embedding=vec_b)

        # Search closer to vec_a
        query = [0.9] + [0.1] + [0.0] * 1022
        results = db.search_similar(query, limit=2)

        assert len(results) == 2
        assert results[0]["filename"] == "a.pdf"

    def test_search_empty_db(self, db):
        query = [0.0] * 1024
        results = db.search_similar(query)
        assert results == []


class TestExecuteSQL:
    def test_valid_select(self, db):
        db.save_invoice(_make_processed())
        results = db.execute_sql("SELECT vendor_name, total_amount FROM invoices")

        assert len(results) == 1
        assert results[0]["vendor_name"] == "Acme Corp"
        assert results[0]["total_amount"] == 1650.0

    def test_rejects_insert(self, db):
        with pytest.raises(ValueError, match="Only SELECT"):
            db.execute_sql("INSERT INTO invoices (filename) VALUES ('x')")

    def test_rejects_delete(self, db):
        with pytest.raises(ValueError, match="Only SELECT"):
            db.execute_sql("DELETE FROM invoices")

    def test_rejects_drop(self, db):
        with pytest.raises(ValueError, match="Only SELECT"):
            db.execute_sql("DROP TABLE invoices")


class TestStats:
    def test_empty_stats(self, db):
        stats = db.get_stats()
        assert stats["count"] == 0
        assert stats["total_spend"] == 0
        assert stats["vendor_count"] == 0

    def test_populated_stats(self, db):
        db.save_invoice(_make_processed("a.pdf", total_amount=100.0))
        db.save_invoice(_make_processed("b.pdf", total_amount=200.0, vendor_name="Other"))

        stats = db.get_stats()
        assert stats["count"] == 2
        assert stats["total_spend"] == 300.0
        assert stats["vendor_count"] == 2
        assert stats["avg_amount"] == 150.0


class TestIsEmpty:
    def test_empty_on_init(self, db):
        assert db.is_empty is True

    def test_not_empty_after_save(self, db):
        db.save_invoice(_make_processed())
        assert db.is_empty is False


class TestClearAll:
    def test_clear(self, db):
        db.save_invoice(_make_processed("a.pdf"), embedding=[0.1] * 1024)
        db.save_invoice(_make_processed("b.pdf"))
        db.clear_all()

        assert db.is_empty
        assert db.get_all_invoices() == []


class TestSerializeF32:
    def test_roundtrip(self):
        vec = [1.0, 2.0, 3.0]
        serialized = _serialize_f32(vec)
        assert isinstance(serialized, bytes)
        assert len(serialized) == 12  # 3 floats * 4 bytes
