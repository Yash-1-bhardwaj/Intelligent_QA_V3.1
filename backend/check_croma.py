
from pathlib import Path
import chromadb

BASE_DIR = Path(__file__).resolve().parent.parent
client = chromadb.PersistentClient(
    path=str(BASE_DIR / "chroma_db")
)

for name in ["page_images", "page_text", "page_descriptions"]:
    print(f"\nTesting {name}...")

    try:
        collection = client.get_collection(name)
        print("Count:", collection.count())

        if collection.count() > 0:
            sample = collection.get(
                limit=1,
                include=["embeddings"]
            )

            embedding = sample["embeddings"][0]

            result = collection.query(
                query_embeddings=[embedding.tolist()],
                n_results=1
            )

            print("Query successful:", result["ids"][0])

    except Exception as e:
        print("FAILED:", repr(e))
