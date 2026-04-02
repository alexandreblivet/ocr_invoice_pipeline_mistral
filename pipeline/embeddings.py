import logging

from mistralai.client import Mistral

from pipeline.schemas import ProcessedInvoice

logger = logging.getLogger(__name__)

EMBED_MODEL = "mistral-embed"
EMBED_DIMENSION = 1024


def embed_text(text: str, api_key: str) -> list[float]:
    """Embed a single text string. Returns 1024-dim float vector."""
    client = Mistral(api_key=api_key)
    response = client.embeddings.create(
        model=EMBED_MODEL,
        inputs=[text],
    )
    return response.data[0].embedding


def embed_texts(texts: list[str], api_key: str) -> list[list[float]]:
    """Embed multiple texts in a single API call."""
    client = Mistral(api_key=api_key)
    response = client.embeddings.create(
        model=EMBED_MODEL,
        inputs=texts,
    )
    sorted_data = sorted(response.data, key=lambda d: d.index)
    return [d.embedding for d in sorted_data]


def build_invoice_chunk(invoice: ProcessedInvoice, raw_markdown: str) -> str:
    """Build a rich text representation for embedding.

    Combines structured fields with raw OCR text so vector search
    can match both precise field lookups and semantic queries.
    """
    data = invoice.data
    items_text = "; ".join(
        f"{item.description} (qty:{item.quantity}, ${item.total})"
        for item in data.line_items
    )
    return (
        f"Invoice {data.invoice_number} from {data.vendor_name}. "
        f"Date: {data.invoice_date}. "
        f"Total: {data.currency} {data.total_amount}. "
        f"Items: {items_text}. "
        f"Payment terms: {data.payment_terms or 'not specified'}. "
        f"Full text: {raw_markdown[:1500]}"
    )
