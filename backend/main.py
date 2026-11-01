from pathlib import Path
import shutil

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from pdf_renderer import render_pdf
from ingestion import DocumentIngestion
from qa_pipeline import QAPipeline
from text_embedding import TextEmbedder
from text_vector_store import TextVectorStore
from page_text_extractor import PageTextExtractor
from vector_store import VectorStore
from text_vector_store import TextVectorStore
from summarizer import DocumentSummarizer

BASE_DIR = Path(__file__).resolve().parent.parent

UPLOAD_DIR = BASE_DIR / "uploads"

UPLOAD_DIR.mkdir(exist_ok=True)


app = FastAPI(
    title="Intelligent Multimodal Document QA"
)
pipeline = QAPipeline(BASE_DIR)
summarizer = DocumentSummarizer()
class QuestionRequest(BaseModel):

    question: str


# --------------------------------
# Upload PDF
# --------------------------------

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):

    pdf_path = UPLOAD_DIR / Path(file.filename).name

    with open(pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 1. Clear and recreate page images
    page_dir = BASE_DIR / "backend" / "page_images"

    if page_dir.exists():
        shutil.rmtree(page_dir)

    page_dir.mkdir(parents=True)

    # 2. Render the newly uploaded PDF
    render_pdf(
        str(pdf_path),
        output_dir=str(page_dir)
    )

    # 3. Reset BOTH vector collections
    ingestion = DocumentIngestion(BASE_DIR)
    ingestion.vector_store.reset()
    ingestion.ingest()

    text_store = TextVectorStore(
        db_path=str(BASE_DIR / "chroma_db")
    )
    text_store.reset()

    # 4. Extract and index text from the SAME PDF
    extractor = PageTextExtractor()
    text_embedder = TextEmbedder()

    extracted_pages = extractor.extract(str(pdf_path))

    for page in extracted_pages:
        text = page["text"]

        # Skip pages without extractable text.
        # Image retrieval will still support these pages.
        if not text.strip():
            continue

        embedding = text_embedder.embed(text)

        text_store.add_page(
            page_id=f"page_{page['page_number']:03d}",
            embedding=embedding,
            metadata={
                "page_number": page["page_number"],
                "text": text
            }
        )
        # Refresh the pipeline after rebuilding the collections
    pipeline.image_store = VectorStore(
        db_path=str(BASE_DIR / "chroma_db")
    )

    pipeline.text_store = TextVectorStore(
        db_path=str(BASE_DIR / "chroma_db")
    )

    return {
        "message": "Document processed successfully",
        "filename": Path(file.filename).name,
        "total_pages": len(extracted_pages),
        "text_pages_indexed": sum(
            1 for page in extracted_pages if page["text"].strip()
        )
    }

# --------------------------------
# Ask question
# --------------------------------

@app.post("/ask")
async def ask_question(
    request: QuestionRequest
):

    result = pipeline.ask(
        request.question
    )


    return result


@app.post("/summarize")
async def summarize_document():
    # Use the PDF most recently uploaded to the uploads directory.
    pdf_files = [
        path for path in UPLOAD_DIR.glob("*.pdf")
        if path.is_file()
    ]

    if not pdf_files:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF before summarizing."
        )

    # main.py's upload endpoint saves the uploaded PDF here.
    latest_pdf = max(
        pdf_files,
        key=lambda path: path.stat().st_mtime
    )

    try:
        summary = summarizer.summarize(str(latest_pdf))
        return {
            "filename": latest_pdf.name,
            "summary": summary
        }
    except Exception as exc:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=500,
            detail=f"Summarization failed: {exc}"
        )
