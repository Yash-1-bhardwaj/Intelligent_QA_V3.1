from pathlib import Path

from multimodal_embedding import MultimodalEmbedder
from vector_store import VectorStore


class DocumentIngestion:

    def __init__(self, base_dir):

        self.base_dir = Path(base_dir)

        self.page_dir = (
            self.base_dir / "backend" / "page_images"
        )

        self.embedder = MultimodalEmbedder()

        self.vector_store = VectorStore(
            db_path=str(self.base_dir / "chroma_db")
        )


    def ingest(self):

        page_images = sorted(
            self.page_dir.glob("page_*.png")
        )

        print(f"Found {len(page_images)} pages")

        for page_number, image_path in enumerate(
            page_images,
            start=1
        ):

            print(
                f"Embedding page "
                f"{page_number}/{len(page_images)}"
            )

            embedding = self.embedder.embed_image(
                image_path
            )

            self.vector_store.add_page(
                page_id=f"page_{page_number:03d}",
                embedding=embedding,
                metadata={
                    "page_number": page_number,
                    "image_path": str(image_path)
                }
            )

        print(
            f"\nStored {len(page_images)} page embeddings."
        )