from streamlit_pdf_viewer import pdf_viewer


def render_pdf(pdf_bytes: bytes, key: str = "pdf_viewer") -> None:
    """Display a PDF in the Streamlit app."""
    pdf_viewer(pdf_bytes, width="100%", height=600, key=key)
