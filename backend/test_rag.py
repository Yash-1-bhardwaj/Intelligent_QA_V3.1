from pathlib import Path

from text_embedding import TextEmbedder
from text_vector_store import TextVectorStore


BASE_DIR = Path(__file__).resolve().parent.parent

embedder = TextEmbedder()

store = TextVectorStore(
    db_path=str(BASE_DIR / "chroma_db")
)

question = "What are Query, Key and Value?"

query_embedding = embedder.embed(question)

results = store.search(
    query_embedding,
    top_k=5
)

print("\nTEXT RAG RESULTS\n")

for i in range(len(results["ids"][0])):

    page_id = results["ids"][0][i]
    distance = results["distances"][0][i]
    metadata = results["metadatas"][0][i]

    print(
        f"{i + 1}. {page_id} "
        f"| distance: {distance:.4f}"
    )

    print(
        f"   Page: {metadata['page_number']}"
    )

    print(
        f"   Text: {metadata['text'][:200]}"
    )

    print()