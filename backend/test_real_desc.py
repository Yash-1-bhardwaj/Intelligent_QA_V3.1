from pathlib import Path

from vlm import VLM
from text_embedding import TextEmbedder
from description_vector_store import DescriptionVectorStore


BASE_DIR = Path(__file__).resolve().parent.parent

PAGE_DIR = BASE_DIR / "backend" / "page_images"

question = (
    "What are the solutions to problems faced "
    "by self-attention?"
)

# Initialize models and vector store
vlm = VLM()
embedder = TextEmbedder()

store = DescriptionVectorStore(
    db_path=str(BASE_DIR / "chroma_db")
)

# Pages we want to test
page_numbers = [15, 16, 17, 18]

for page_number in page_numbers:

    image_path = PAGE_DIR / f"page_{page_number:03d}.png"

    if not image_path.exists():
        print(f"Skipping missing page: {page_number}")
        continue

    print(f"\nAnalyzing page {page_number}...")

    # Generate a question-focused description
    description = vlm.describe_page(
        question,
        str(image_path)
    )

    print("\nDescription:")
    print(description)
    page_id=f"page_{page_number:03d}"
    # Skip pages the VLM considers irrelevant
    if "NOT_RELEVANT" in description.upper():
        store.collection.delete(ids=[page_id])
        print(f"Page {page_number} is not relevant. Removed from store.")
        continue

    # Embed the generated description
    embedding = embedder.embed(description)

    # Store it using the corresponding page ID
    store.add_description(
        page_id=f"page_{page_number:03d}",
        embedding=embedding,
        description=description
    )

    print(f"Stored description for page {page_number}.")

print("\nReal description generation completed.")