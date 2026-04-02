import json
import logging

from mistralai.client import Mistral
from mistralai.client.models import DocumentURLChunk
from mistralai.client.models.responseformat import ResponseFormat, JSONSchema

from pipeline.ocr import _upload_and_get_url, run_ocr, get_markdown
from pipeline.schemas import InvoiceData, ProcessedInvoice

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """Extract structured invoice data from the OCR text below.

LINE ITEMS — extract each row from any itemised table:
- "description": product or service name
- "quantity": number of units (default 1 if absent)
- "unit_price": price per unit
- "total": line total (quantity × unit_price)
If there is no itemised table, create one line item with the total amount.

TOTALS: subtotal = sum of line totals, tax_amount = tax if shown (else null), total_amount = final amount due.
DATES: preserve original format. CURRENCY: ISO 3-letter code (USD, EUR, etc.).
Use null for optional fields not clearly present."""

ANNOTATION_PROMPT = """Extract structured invoice data into the JSON schema provided.

LINE ITEMS — read each row from the invoice table:
- "description": the product or service name (plain text, no nested objects)
- "quantity": number of units (default 1 if not listed)
- "unit_price": price per single unit
- "total": line total (quantity × unit_price)
If the invoice has no itemised table, create one line item using the total amount.

TOTALS: subtotal is the sum of line item totals. tax_amount is the tax if shown, else null. total_amount is the final amount due.

DATES: preserve the original format from the document.
CURRENCY: use ISO 3-letter code (USD, EUR, GBP, etc.).
Use null for any optional field not clearly present in the document."""


def _inline_schema() -> dict:
    """Return InvoiceData JSON schema with $ref/$defs resolved inline.

    Mistral's document_annotation_format does not follow JSON $ref pointers,
    so we dereference them before sending the schema.
    """
    schema = InvoiceData.model_json_schema()
    defs = schema.pop("$defs", {})
    if not defs:
        return schema

    def _resolve(node):
        if isinstance(node, dict):
            if "$ref" in node:
                ref_name = node["$ref"].rsplit("/", 1)[-1]
                return _resolve(defs[ref_name])
            return {k: _resolve(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_resolve(item) for item in node]
        return node

    return _resolve(schema)


def extract_with_annotation(pdf_bytes: bytes, api_key: str, filename: str = "document.pdf") -> tuple[InvoiceData, str]:
    """Tier 1: Use document_annotation_format for single-call structured extraction."""
    client = Mistral(api_key=api_key)
    url = _upload_and_get_url(client, pdf_bytes, filename)
    logger.info("Tier 1: annotation extraction for %s", filename)
    response = client.ocr.process(
        model="mistral-ocr-latest",
        document=DocumentURLChunk(document_url=url),
        table_format="html",
        include_image_base64=True,
        document_annotation_format=ResponseFormat(
            type="json_schema",
            json_schema=JSONSchema(
                name="InvoiceData",
                schema_definition=_inline_schema(),
            ),
        ),
        document_annotation_prompt=ANNOTATION_PROMPT,
    )
    markdown = get_markdown(response)

    annotation = response.document_annotation
    if annotation is None:
        raise ValueError("No document annotation returned")

    if isinstance(annotation, str):
        raw = json.loads(annotation)
    elif isinstance(annotation, dict):
        raw = annotation
    elif isinstance(annotation, InvoiceData):
        return annotation, markdown
    else:
        raw = json.loads(str(annotation))

    # Mistral sometimes wraps extracted data inside the JSON Schema structure
    if "properties" in raw and "vendor_name" not in raw:
        raw = raw["properties"]

    data = InvoiceData.model_validate(raw)

    logger.info("Tier 1 succeeded for %s: %s / %s", filename, data.vendor_name, data.invoice_number)
    return data, markdown


def extract_with_chat(markdown: str, api_key: str) -> InvoiceData:
    """Tier 2 fallback: OCR markdown -> chat.parse() with Pydantic response_format."""
    client = Mistral(api_key=api_key)
    logger.info("Tier 2: chat.parse extraction")
    response = client.chat.parse(
        model="mistral-small-latest",
        messages=[
            {"role": "system", "content": EXTRACTION_PROMPT},
            {"role": "user", "content": markdown},
        ],
        response_format=InvoiceData,
        temperature=0,
    )
    return response.choices[0].message.parsed


def process_invoice(pdf_bytes: bytes, filename: str, api_key: str) -> ProcessedInvoice:
    """Process a single invoice PDF through the extraction pipeline."""
    # Tier 1: annotation-based extraction
    tier1_error = None
    try:
        data, markdown = extract_with_annotation(pdf_bytes, api_key, filename)
        return ProcessedInvoice(
            filename=filename,
            data=data,
            raw_markdown=markdown,
            extraction_method="annotation",
        )
    except Exception as e:
        tier1_error = e
        logger.warning("Tier 1 failed for %s: %s", filename, e)

    # Tier 2: OCR + chat.parse fallback
    try:
        ocr_response = run_ocr(pdf_bytes, api_key, filename)
        markdown = get_markdown(ocr_response)
        data = extract_with_chat(markdown, api_key)
        logger.info("Tier 2 succeeded for %s", filename)
        return ProcessedInvoice(
            filename=filename,
            data=data,
            raw_markdown=markdown,
            extraction_method="chat_parse",
        )
    except Exception as e:
        logger.error("Both extraction tiers failed for %s", filename)
        raise RuntimeError(
            f"Failed to extract invoice data from {filename}. "
            f"Tier 1 (annotation): {tier1_error}. "
            f"Tier 2 (chat.parse): {e}"
        ) from e
