import pytest
from pydantic import ValidationError

from pipeline.schemas import LineItem, InvoiceData, ProcessedInvoice


# --- LineItem ---

class TestLineItem:
    def test_valid(self):
        item = LineItem(description="Widget", quantity=2, unit_price=10.0, total=20.0)
        assert item.description == "Widget"
        assert item.quantity == 2.0
        assert item.unit_price == 10.0
        assert item.total == 20.0

    def test_int_coerced_to_float(self):
        item = LineItem(description="Item", quantity=1, unit_price=5, total=5)
        assert isinstance(item.quantity, float)
        assert isinstance(item.unit_price, float)

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            LineItem(description="Widget", quantity=2, unit_price=10.0)

    def test_zero_values(self):
        item = LineItem(description="Free item", quantity=0, unit_price=0, total=0)
        assert item.total == 0.0

    def test_negative_values_allowed(self):
        item = LineItem(description="Discount", quantity=1, unit_price=-5.0, total=-5.0)
        assert item.total == -5.0


# --- InvoiceData ---

MINIMAL_INVOICE = dict(
    vendor_name="Acme Corp",
    invoice_number="INV-001",
    invoice_date="2025-01-15",
    currency="USD",
    line_items=[{"description": "Service", "quantity": 1, "unit_price": 100, "total": 100}],
    subtotal=100.0,
    total_amount=100.0,
)


class TestInvoiceData:
    def test_minimal_valid(self):
        inv = InvoiceData(**MINIMAL_INVOICE)
        assert inv.vendor_name == "Acme Corp"
        assert inv.vendor_address is None
        assert inv.due_date is None
        assert inv.tax_amount is None
        assert inv.payment_terms is None
        assert len(inv.line_items) == 1

    def test_full_fields(self):
        inv = InvoiceData(
            **MINIMAL_INVOICE,
            vendor_address="123 Main St",
            due_date="2025-02-15",
            tax_amount=10.0,
            payment_terms="Net 30",
        )
        assert inv.vendor_address == "123 Main St"
        assert inv.tax_amount == 10.0

    def test_missing_vendor_name_raises(self):
        data = {**MINIMAL_INVOICE}
        del data["vendor_name"]
        with pytest.raises(ValidationError) as exc_info:
            InvoiceData(**data)
        assert "vendor_name" in str(exc_info.value)

    def test_missing_line_items_raises(self):
        data = {**MINIMAL_INVOICE}
        del data["line_items"]
        with pytest.raises(ValidationError):
            InvoiceData(**data)

    def test_empty_line_items_allowed(self):
        data = {**MINIMAL_INVOICE, "line_items": []}
        inv = InvoiceData(**data)
        assert inv.line_items == []

    def test_multiple_line_items(self):
        items = [
            {"description": "A", "quantity": 1, "unit_price": 50, "total": 50},
            {"description": "B", "quantity": 2, "unit_price": 25, "total": 50},
        ]
        inv = InvoiceData(**{**MINIMAL_INVOICE, "line_items": items})
        assert len(inv.line_items) == 2
        assert inv.line_items[1].quantity == 2.0

    def test_json_roundtrip(self):
        inv = InvoiceData(**MINIMAL_INVOICE)
        json_str = inv.model_dump_json()
        restored = InvoiceData.model_validate_json(json_str)
        assert restored == inv

    def test_json_schema_generation(self):
        schema = InvoiceData.model_json_schema()
        assert "properties" in schema
        assert "vendor_name" in schema["properties"]
        assert "line_items" in schema["properties"]


# --- ProcessedInvoice ---

class TestProcessedInvoice:
    def test_defaults(self):
        inv_data = InvoiceData(**MINIMAL_INVOICE)
        proc = ProcessedInvoice(filename="test.pdf", data=inv_data)
        assert proc.raw_markdown == ""
        assert proc.extraction_method == "annotation"

    def test_chat_parse_method(self):
        inv_data = InvoiceData(**MINIMAL_INVOICE)
        proc = ProcessedInvoice(
            filename="test.pdf",
            data=inv_data,
            raw_markdown="# Invoice",
            extraction_method="chat_parse",
        )
        assert proc.extraction_method == "chat_parse"

    def test_nested_serialization(self):
        inv_data = InvoiceData(**MINIMAL_INVOICE)
        proc = ProcessedInvoice(filename="test.pdf", data=inv_data)
        d = proc.model_dump()
        assert d["data"]["vendor_name"] == "Acme Corp"
        assert d["data"]["line_items"][0]["description"] == "Service"
