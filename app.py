import os
import json
import glob as glob_mod
from io import StringIO

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from mistralai.client import Mistral

from pipeline.extract import process_invoice
from pipeline.schemas import InvoiceData, LineItem, ProcessedInvoice
from components.pdf_viewer import render_pdf

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Invoice Processor",
    page_icon="\U0001f4c4",
    layout="wide",
)

# ── Custom CSS (Mistral orange accent) ───────────────────────────────────────

st.markdown(
    """
    <style>
    :root { --accent: #FF7000; }
    div.stButton > button[kind="primary"],
    div.stDownloadButton > button[kind="primary"] {
        background-color: #FF7000;
        border-color: #FF7000;
        color: white;
    }
    div.stButton > button[kind="primary"]:hover,
    div.stDownloadButton > button[kind="primary"]:hover {
        background-color: #e06400;
        border-color: #e06400;
    }
    div[data-baseweb="tab-highlight"] { background-color: #FF7000 !important; }
    .stProgress > div > div > div { background-color: #FF7000; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state defaults ───────────────────────────────────────────────────

_DEFAULTS: dict = {
    "processed_invoices": {},  # filename -> ProcessedInvoice dict
    "chat_history": [],
    "uploaded_files_cache": {},  # filename -> bytes
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Helpers ──────────────────────────────────────────────────────────────────


def get_api_key() -> str | None:
    return os.environ.get("MISTRAL_API_KEY") or st.session_state.get("_api_key")


def _process_and_store(filename: str, pdf_bytes: bytes, api_key: str) -> bool:
    """Process a single file and store in session state. Returns True on success."""
    if filename in st.session_state.processed_invoices:
        return True
    try:
        result = process_invoice(pdf_bytes, filename, api_key)
        st.session_state.processed_invoices[filename] = result.model_dump()
        st.session_state.uploaded_files_cache[filename] = pdf_bytes
        return True
    except Exception as e:
        st.error(f"Failed to process **{filename}**: {e}")
        return False


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Invoice Processor")
    st.caption("Powered by Mistral OCR 3")
    st.divider()

    # API key
    env_key = os.environ.get("MISTRAL_API_KEY", "")
    if not env_key:
        st.text_input(
            "Mistral API Key",
            type="password",
            key="_api_key",
            help="Set MISTRAL_API_KEY in .env or enter here",
        )

    api_key = get_api_key()
    if not api_key:
        st.warning("Enter your Mistral API key to get started.")

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload invoices",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    # Sample data button
    sample_dir = os.path.join(os.path.dirname(__file__), "samples")
    sample_files = sorted(glob_mod.glob(os.path.join(sample_dir, "*.pdf")))

    if sample_files and st.button("Try with sample data", type="primary"):
        if not api_key:
            st.error("Set an API key first.")
        else:
            progress = st.progress(0, text="Loading samples...")
            for i, path in enumerate(sample_files):
                name = os.path.basename(path)
                with open(path, "rb") as f:
                    pdf_bytes = f.read()
                _process_and_store(name, pdf_bytes, api_key)
                progress.progress((i + 1) / len(sample_files), text=f"Processed {name}")
            progress.empty()
            st.rerun()

    # Status
    n = len(st.session_state.processed_invoices)
    if n:
        st.divider()
        st.metric("Processed invoices", n)
        for fname in st.session_state.processed_invoices:
            st.caption(f"- {fname}")

# ── Process uploaded files ───────────────────────────────────────────────────

if uploaded_files and api_key:
    new_files = [
        f for f in uploaded_files if f.name not in st.session_state.processed_invoices
    ]
    if new_files:
        progress = st.progress(0, text="Processing invoices...")
        for i, uf in enumerate(new_files):
            pdf_bytes = uf.read()
            _process_and_store(uf.name, pdf_bytes, api_key)
            st.session_state.uploaded_files_cache[uf.name] = pdf_bytes
            progress.progress((i + 1) / len(new_files), text=f"Processing {uf.name}...")
        progress.empty()
        st.rerun()
    else:
        # Cache bytes for already-processed files so PDF viewer works
        for uf in uploaded_files:
            if uf.name not in st.session_state.uploaded_files_cache:
                st.session_state.uploaded_files_cache[uf.name] = uf.read()

# ── Tabs ─────────────────────────────────────────────────────────────────────

tab1, tab2 = st.tabs(["\U0001f4cb Validation", "\U0001f4ac Chat over Invoices"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — VALIDATION
# ════════════════════════════════════════════════════════════════════════════

with tab1:
    invoices = st.session_state.processed_invoices

    if not invoices:
        st.info("Upload invoices in the sidebar or click **Try with sample data** to get started.")
    else:
        # ── Batch summary table ──────────────────────────────────────────
        if len(invoices) > 1:
            st.subheader("Batch Summary")
            summary_rows = []
            for fname, inv in invoices.items():
                d = inv["data"]
                summary_rows.append(
                    {
                        "File": fname,
                        "Vendor": d["vendor_name"],
                        "Invoice #": d["invoice_number"],
                        "Date": d["invoice_date"],
                        "Total": d["total_amount"],
                        "Currency": d["currency"],
                        "Method": inv["extraction_method"],
                    }
                )
            summary_df = pd.DataFrame(summary_rows)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            # Batch CSV export
            csv_buf = StringIO()
            all_rows = []
            for fname, inv in invoices.items():
                d = inv["data"]
                for item in d["line_items"]:
                    row = {
                        "file": fname,
                        "vendor": d["vendor_name"],
                        "invoice_number": d["invoice_number"],
                        "invoice_date": d["invoice_date"],
                        "currency": d["currency"],
                        **item,
                    }
                    all_rows.append(row)
            if all_rows:
                batch_df = pd.DataFrame(all_rows)
                st.download_button(
                    "Export All to CSV",
                    batch_df.to_csv(index=False),
                    "all_invoices.csv",
                    "text/csv",
                    type="primary",
                )
            st.divider()

        # ── Invoice selector ─────────────────────────────────────────────
        filenames = list(invoices.keys())
        if len(filenames) == 1:
            selected = filenames[0]
        else:
            selected = st.selectbox("Select invoice to view", filenames)

        inv_dict = invoices[selected]
        inv_data = inv_dict["data"]
        method = inv_dict["extraction_method"]

        # Status banner
        if method == "annotation":
            st.success(f"Extracted via document annotation (high confidence)")
        else:
            st.warning(f"Extracted via chat fallback — please verify the results")

        # ── Two-column layout ────────────────────────────────────────────
        left, right = st.columns([1, 1])

        with left:
            st.subheader("Document Preview")
            pdf_bytes = st.session_state.uploaded_files_cache.get(selected)
            if pdf_bytes:
                render_pdf(pdf_bytes, key=f"pdf_{selected}")
            else:
                st.info("PDF preview not available (file was loaded in a previous session).")

        with right:
            st.subheader("Extracted Data")

            # Editable fields
            c1, c2 = st.columns(2)
            vendor_name = c1.text_input("Vendor Name", value=inv_data["vendor_name"], key=f"vn_{selected}")
            invoice_number = c2.text_input("Invoice Number", value=inv_data["invoice_number"], key=f"in_{selected}")

            c3, c4 = st.columns(2)
            invoice_date = c3.text_input("Invoice Date", value=inv_data["invoice_date"], key=f"id_{selected}")
            due_date = c4.text_input("Due Date", value=inv_data.get("due_date") or "", key=f"dd_{selected}")

            vendor_address = st.text_input(
                "Vendor Address", value=inv_data.get("vendor_address") or "", key=f"va_{selected}"
            )
            payment_terms = st.text_input(
                "Payment Terms", value=inv_data.get("payment_terms") or "", key=f"pt_{selected}"
            )

            currency = st.text_input("Currency", value=inv_data["currency"], key=f"cur_{selected}")

            # Line items
            st.markdown("**Line Items**")
            items_data = [
                {
                    "description": it["description"],
                    "quantity": it["quantity"],
                    "unit_price": it["unit_price"],
                    "total": it["total"],
                }
                for it in inv_data["line_items"]
            ]
            if not items_data:
                items_data = [{"description": "", "quantity": 0.0, "unit_price": 0.0, "total": 0.0}]
            items_df = pd.DataFrame(items_data)
            edited_items = st.data_editor(
                items_df,
                num_rows="dynamic",
                use_container_width=True,
                key=f"items_{selected}",
            )

            # Totals
            c5, c6, c7 = st.columns(3)
            subtotal = c5.number_input("Subtotal", value=float(inv_data["subtotal"]), key=f"sub_{selected}")
            tax_amount = c6.number_input(
                "Tax Amount", value=float(inv_data.get("tax_amount") or 0.0), key=f"tax_{selected}"
            )
            total_amount = c7.number_input(
                "Total Amount", value=float(inv_data["total_amount"]), key=f"tot_{selected}"
            )

            # Build updated invoice from current widget values
            def _build_updated_invoice() -> InvoiceData:
                line_items = []
                for _, row in edited_items.iterrows():
                    line_items.append(
                        LineItem(
                            description=str(row["description"]),
                            quantity=float(row["quantity"]),
                            unit_price=float(row["unit_price"]),
                            total=float(row["total"]),
                        )
                    )
                return InvoiceData(
                    vendor_name=vendor_name,
                    vendor_address=vendor_address or None,
                    invoice_number=invoice_number,
                    invoice_date=invoice_date,
                    due_date=due_date or None,
                    currency=currency,
                    line_items=line_items,
                    subtotal=subtotal,
                    tax_amount=tax_amount or None,
                    total_amount=total_amount,
                    payment_terms=payment_terms or None,
                )

            # Export buttons
            st.divider()
            ec1, ec2 = st.columns(2)
            updated = _build_updated_invoice()
            with ec1:
                st.download_button(
                    "Export JSON",
                    updated.model_dump_json(indent=2),
                    f"{selected.rsplit('.', 1)[0]}.json",
                    "application/json",
                    type="primary",
                )
            with ec2:
                export_rows = [item.model_dump() for item in updated.line_items]
                if export_rows:
                    csv_export = pd.DataFrame(export_rows).to_csv(index=False)
                else:
                    csv_export = ""
                st.download_button(
                    "Export CSV",
                    csv_export,
                    f"{selected.rsplit('.', 1)[0]}_items.csv",
                    "text/csv",
                    type="primary",
                )

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — CHAT OVER INVOICES
# ════════════════════════════════════════════════════════════════════════════

with tab2:
    invoices = st.session_state.processed_invoices
    n_inv = len(invoices)

    if not n_inv:
        st.info(
            "No invoices loaded yet. Upload invoices in the sidebar or use "
            "**Try with sample data**, then come back here to chat."
        )
    else:
        st.markdown(f"**Chatting over {n_inv} invoice{'s' if n_inv != 1 else ''}**")

        # Build context
        context_parts = []
        for fname, inv in invoices.items():
            context_parts.append(f"## {fname}\n```json\n{json.dumps(inv['data'], indent=2)}\n```")
        invoices_context = "\n\n".join(context_parts)

        system_prompt = (
            "You are an invoice analysis assistant. Here are the processed invoices:\n\n"
            f"{invoices_context}\n\n"
            "Answer the user's question based on this data. Be precise with numbers. "
            "If you calculate totals, show your work. Use markdown formatting."
        )

        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Example query chips (only when history is empty)
        if not st.session_state.chat_history:
            st.markdown("**Try asking:**")
            examples = [
                "Total spend across all invoices?",
                "Which invoices are overdue?",
                "Break down spending by vendor",
                "Find the most expensive line item",
            ]
            cols = st.columns(len(examples))
            for col, example in zip(cols, examples):
                if col.button(example, key=f"ex_{example[:20]}"):
                    st.session_state.chat_history.append({"role": "user", "content": example})
                    st.rerun()

        # Chat input
        if prompt := st.chat_input("Ask about your invoices..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.rerun()

        # Generate response for last user message if no assistant reply yet
        if (
            st.session_state.chat_history
            and st.session_state.chat_history[-1]["role"] == "user"
            and api_key
        ):
            with st.chat_message("user"):
                st.markdown(st.session_state.chat_history[-1]["content"])

            with st.chat_message("assistant"):
                placeholder = st.empty()
                full_response = ""

                messages = [{"role": "system", "content": system_prompt}]
                messages.extend(
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.chat_history
                )

                try:
                    client = Mistral(api_key=api_key)
                    stream = client.chat.stream(
                        model="mistral-large-latest",
                        messages=messages,
                    )
                    for event in stream:
                        token = event.data.choices[0].delta.content
                        if token:
                            full_response += token
                            placeholder.markdown(full_response + "\u258c")
                    placeholder.markdown(full_response)
                except Exception as e:
                    full_response = f"Error generating response: {e}"
                    placeholder.error(full_response)

                st.session_state.chat_history.append(
                    {"role": "assistant", "content": full_response}
                )
