from pathlib import Path

from text_embedding import TextEmbedder
from description_vector_store import DescriptionVectorStore


BASE_DIR = Path(__file__).resolve().parent.parent

embedder = TextEmbedder()

store = DescriptionVectorStore(
    db_path=str(BASE_DIR / "chroma_db")
)

question = (
    "What are the solutions to problems faced "
    "by self-attention?"
)

query_embedding = embedder.embed(question)

results = store.search(
    query_embedding,
    top_k=4
)

print("\nREAL DESCRIPTION RAG RESULTS\n")

for i, page_id in enumerate(results["ids"][0]):

    print(
        f"Rank: {i + 1}"
        f" | Page: {results['metadatas'][0][i]['page_number']}"
        f" | Distance: {results['distances'][0][i]:.4f}"
    )

    print(
        results["documents"][0][i][:300]
    )

    print()