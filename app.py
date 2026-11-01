
from pathlib import Path
import requests
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BASE_DIR / "backend"
PAGE_DIR = BACKEND_DIR / "page_images"
API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Intelligent Multimodal Document QA",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Intelligent Multimodal Document QA")
st.write("Upload a PDF to ask questions or generate a text-based summary.")

# Initialize session state
defaults = {
    "document_processed": False,
    "uploaded_filename": "",
    "document_summary": "",
    "summary_filename": "",
    "answer": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# -------------------------
# PDF UPLOAD AND PROCESSING
# -------------------------
st.header("1. Upload Document")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)

if uploaded_file and st.button("Process PDF", type="primary"):
    with st.spinner("Processing PDF and creating embeddings..."):
        try:
            response = requests.post(
                f"{API_URL}/upload",
                files={
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "application/pdf"
                    )
                },
                timeout=None
            )

            if response.ok:
                data = response.json()

                st.session_state.document_processed = True
                st.session_state.uploaded_filename = uploaded_file.name

                # Clear results from the previous document
                st.session_state.document_summary = ""
                st.session_state.summary_filename = ""
                st.session_state.answer = None

                st.success(
                    f"Processed {uploaded_file.name} successfully!"
                )
                st.write(
                    f"Total pages: {data.get('total_pages', 'Unknown')}"
                )
            else:
                st.error(f"PDF processing failed: {response.text}")

        except requests.RequestException as exc:
            st.error(f"Could not connect to FastAPI: {exc}")

# -------------------------
# TEXT-BASED PDF SUMMARY
# -------------------------
if st.session_state.document_processed:
    st.divider()
    st.header("2. Text-Based PDF Summary")

    st.caption(
        "Generates a summary from extractable PDF text. "
        "Content present only in images or diagrams may be missed."
    )

    if st.button("📝 Generate Text-Based PDF Summary"):
        with st.spinner(
            "Generating summary from the PDF text... This may take a while."
        ):
            try:
                response = requests.post(
                    f"{API_URL}/summarize",
                    timeout=None
                )

                if response.ok:
                    data = response.json()
                    st.session_state.document_summary = data["summary"]
                    st.session_state.summary_filename = data["filename"]
                else:
                    st.error(f"Summarization failed: {response.text}")

            except requests.RequestException as exc:
                st.error(f"Could not connect to FastAPI: {exc}")

    if st.session_state.document_summary:
        st.subheader(
            f"Summary: {st.session_state.summary_filename}"
        )
        st.markdown(st.session_state.document_summary)

# -------------------------
# QUESTION ANSWERING
# -------------------------
if st.session_state.document_processed:
    st.divider()
    st.header("3. Ask Questions")

    question = st.text_area(
        "Enter your question about the PDF:",
        placeholder="e.g. Explain the main concepts in this document."
    )

    if st.button("Get Answer", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Finding relevant pages and generating answer..."):
                try:
                    response = requests.post(
                        f"{API_URL}/ask",
                        json={"question": question.strip()},
                        timeout=None
                    )

                    if response.ok:
                        st.session_state.answer = response.json()
                    else:
                        st.error(f"Question answering failed: {response.text}")

                except requests.RequestException as exc:
                    st.error(f"Could not connect to FastAPI: {exc}")

    # Display the most recent answer
    result = st.session_state.answer

    if result:
        st.subheader("Answer")
        st.markdown(result.get("answer", "No answer returned."))

        pages = result.get("pages", [])

        if pages:
            st.subheader("Source Pages")

            columns = st.columns(min(len(pages), 3))

            for index, page_number in enumerate(pages):
                image_path = PAGE_DIR / f"page_{page_number:03d}.png"

                with columns[index % len(columns)]:
                    st.caption(f"Page {page_number}")

                    if image_path.exists():
                        st.image(
                            str(image_path),
                            use_container_width=True
                        )
                    else:
                        st.warning("Source page image not found.")
