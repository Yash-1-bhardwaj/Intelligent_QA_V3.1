
from pathlib import Path

from description_vector_store import DescriptionVectorStore

BASE_DIR = Path(__file__).resolve().parent.parent

store = DescriptionVectorStore(
    db_path=str(BASE_DIR / "chroma_db")
)

results = store.collection.get(
    include=["documents", "metadatas"]
)

print("\nSTORED PAGE DESCRIPTIONS\n")

for page_id, document, metadata in zip(
    results["ids"],
    results["documents"],
    results["metadatas"]
):
    print(f"\nPage ID: {page_id}")
    print(f"Page number: {metadata.get('page_number')}")
    print(f"Description: {document[:250]}")
