from pathlib import Path

from page_text_extractor import PageTextExtractor
from text_embedding import TextEmbedder
from text_vector_store import TextVectorStore


BASE_DIR = Path(__file__).resolve().parent.parent


extractor = PageTextExtractor()
embedder = TextEmbedder()

store = TextVectorStore(
    db_path=str(BASE_DIR / "chroma_db")
)

store.reset()


pages = extractor.extract(
    BASE_DIR / "document" / "Transformer.pdf"
)

for page in pages:

    text = page["text"]

    # Skip completely empty pages
    if not text:
        continue

    embedding = embedder.embed(text)

    store.add_page(
        page_id=f"page_{page['page_number']:03d}",
        embedding=embedding,
        metadata={
            "page_number": page["page_number"],
            "text": text
        }
    )

print("Stored text embeddings successfully.")