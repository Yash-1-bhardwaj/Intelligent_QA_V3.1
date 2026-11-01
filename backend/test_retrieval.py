from pathlib import Path

from multimodal_embedding import MultimodalEmbedder
from vector_store import VectorStore


BASE_DIR = Path(__file__).resolve().parent.parent


embedder = MultimodalEmbedder()

vector_store = VectorStore(
    db_path=str(BASE_DIR / "chroma_db")
)


question = "What are Query, Key and Value?"

query_embedding = embedder.embed_text(question)

results = vector_store.search(
    query_embedding,
    top_k=5
)


print("\n==============================")
print("RETRIEVED PAGES")
print("==============================")

for i in range(len(results["ids"][0])):

    page_id = results["ids"][0][i]
    metadata = results["metadatas"][0][i]
    distance = results["distances"][0][i]

    print(
        f"{i + 1}. {page_id} "
        f"| distance: {distance:.4f} "
        f"| image: {metadata['image_path']}"
    )