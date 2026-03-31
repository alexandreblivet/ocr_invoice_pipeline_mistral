from pydantic import BaseModel, Field
from typing import Optional


class LineItem(BaseModel):
    description: str = Field(description="Description of the item or service")
    quantity: float = Field(description="Quantity of items")
    unit_price: float = Field(description="Price per unit")
    total: float = Field(description="Total amount for this line item")


class InvoiceData(BaseModel):
    vendor_name: str = Field(description="Name of the vendor or supplier")
    vendor_address: Optional[str] = Field(default=None, description="Full address of the vendor")
    invoice_number: str = Field(description="Unique invoice identifier")
    invoice_date: str = Field(description="Date the invoice was issued (any format)")
    due_date: Optional[str] = Field(default=None, description="Payment due date")
    currency: str = Field(description="Currency code, e.g. USD, EUR, GBP")
    line_items: list[LineItem] = Field(description="List of billed items or services")
    subtotal: float = Field(description="Sum before tax")
    tax_amount: Optional[float] = Field(default=None, description="Total tax amount")
    total_amount: float = Field(description="Final total amount due")
    payment_terms: Optional[str] = Field(default=None, description="Payment terms or conditions")


class ProcessedInvoice(BaseModel):
    filename: str
    data: InvoiceData
    raw_markdown: str = ""
    extraction_method: str = "annotation"
