# Invoice Processor

A Streamlit application that extracts structured data from invoice PDFs using Mistral's Document AI stack. Upload invoices, review and edit extracted fields side-by-side with the original document, export to JSON/CSV, and chat with an AI assistant about your invoice data.

## Setup

```bash
# Clone and install dependencies
cd ocr_invoice_pipeline_mistral
uv sync

# Configure API key
cp .env.example .env
# Edit .env and add your Mistral API key

# Run the app
uv run streamlit run app.py
```

## Architecture

```
app.py                  Streamlit UI — two tabs (Validation + Chat)
pipeline/
  schemas.py            Pydantic models: LineItem, InvoiceData, ProcessedInvoice
  ocr.py                Mistral OCR 3 wrapper (base64 PDF → markdown)
  extract.py            Two-tier extraction:
                          1. document_annotation_format (single API call)
                          2. chat.parse() fallback (OCR → structured output)
components/
  pdf_viewer.py         streamlit-pdf-viewer wrapper
samples/                Sample invoice PDFs for demo
```

### Extraction pipeline

1. **OCR**: `mistral-ocr-latest` extracts raw text/tables from the PDF
2. **Structured extraction (Tier 1)**: `document_annotation_format` with a Pydantic schema gets structured JSON directly from the OCR step
3. **Fallback (Tier 2)**: If annotation fails, the OCR markdown is fed to `mistral-small-latest` via `chat.parse()` with the same Pydantic schema as `response_format`
4. **Chat**: Processed invoice data is serialized as context for `mistral-large-latest` to answer natural-language questions

## Screenshots

_Coming soon_

## Built with

- [Mistral OCR 3](https://docs.mistral.ai/capabilities/document_ai/) — document understanding and OCR
- [Mistral Small](https://docs.mistral.ai/getting-started/models/) — structured data extraction
- [Mistral Large](https://docs.mistral.ai/getting-started/models/) — conversational RAG
- [Streamlit](https://streamlit.io/) — web UI
- [streamlit-pdf-viewer](https://pypi.org/project/streamlit-pdf-viewer/) — in-browser PDF display
- [Pydantic](https://docs.pydantic.dev/) — schema validation
