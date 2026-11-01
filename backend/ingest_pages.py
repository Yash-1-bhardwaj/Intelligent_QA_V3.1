from pathlib import Path

from multimodal_embedding import MultimodalEmbedder
from vector_store import VectorStore


BASE_DIR = Path(__file__).resolve().parent.parent

PAGE_DIR = Path(__file__).resolve().parent / "page_images"


embedder = MultimodalEmbedder()

vector_store = VectorStore(
    db_path=str(BASE_DIR / "chroma_db")
)


page_images = sorted(PAGE_DIR.glob("page_*.png"))

print(f"Found {len(page_images)} pages")


for page_number, image_path in enumerate(page_images, start=1):

    print(
        f"Embedding page {page_number}/{len(page_images)}"
    )

    embedding = embedder.embed_image(image_path)

    vector_store.add_page(
        page_id=f"page_{page_number:03d}",
        embedding=embedding,
        metadata={
            "page_number": page_number,
            "image_path": str(image_path)
        }
    )


print("\nDone!")
print(f"Stored {len(page_images)} page embeddings.")