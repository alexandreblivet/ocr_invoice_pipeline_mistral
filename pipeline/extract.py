import json

from mistralai.client import Mistral
from mistralai.client.models import DocumentURLChunk
from mistralai.client.models.responseformat import ResponseFormat, JSONSchema

from pipeline.ocr import _upload_and_get_url, run_ocr, get_markdown
from pipeline.schemas import InvoiceData, ProcessedInvoice

EXTRACTION_PROMPT = """You are an invoice data extraction assistant.
Extract all invoice fields from the following OCR text.
Return ONLY the structured data matching the schema. Be precise with numbers.
If a field is not found, use null for optional fields."""


def extract_with_annotation(pdf_bytes: bytes, api_key: str, filename: str = "document.pdf") -> tuple[InvoiceData, str]:
    """Tier 1: Use document_annotation_format for single-call structured extraction."""
    client = Mistral(api_key=api_key)
    url = _upload_and_get_url(client, pdf_bytes, filename)
    response = client.ocr.process(
        model="mistral-ocr-latest",
        document=DocumentURLChunk(document_url=url),
        table_format="html",
        include_image_base64=True,
        document_annotation_format=ResponseFormat(
            type="json_schema",
            json_schema=JSONSchema(
                name="InvoiceData",
                schema_definition=InvoiceData.model_json_schema(),
            ),
        ),
        document_annotation_prompt="Extract all invoice fields: vendor info, invoice number, dates, line items, totals, and payment terms.",
    )
    markdown = get_markdown(response)

    annotation = response.document_annotation
    if annotation is None:
        raise ValueError("No document annotation returned")

    if isinstance(annotation, str):
        data = InvoiceData.model_validate_json(annotation)
    elif isinstance(annotation, dict):
        data = InvoiceData.model_validate(annotation)
    elif isinstance(annotation, InvoiceData):
        data = annotation
    else:
        data = InvoiceData.model_validate(json.loads(str(annotation)))

    return data, markdown


def extract_with_chat(markdown: str, api_key: str) -> InvoiceData:
    """Tier 2 fallback: OCR markdown → chat.parse() with Pydantic response_format."""
    client = Mistral(api_key=api_key)
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
    try:
        data, markdown = extract_with_annotation(pdf_bytes, api_key, filename)
        return ProcessedInvoice(
            filename=filename,
            data=data,
            raw_markdown=markdown,
            extraction_method="annotation",
        )
    except Exception:
        pass

    # Tier 2: OCR + chat.parse fallback
    try:
        ocr_response = run_ocr(pdf_bytes, api_key, filename)
        markdown = get_markdown(ocr_response)
        data = extract_with_chat(markdown, api_key)
        return ProcessedInvoice(
            filename=filename,
            data=data,
            raw_markdown=markdown,
            extraction_method="chat_parse",
        )
    except Exception as e:
        raise RuntimeError(f"Failed to extract invoice data from {filename}: {e}") from e
