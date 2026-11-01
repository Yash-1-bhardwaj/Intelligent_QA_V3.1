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
descriptions = {
    "page_015": """
    This page introduces the self-attention mechanism
    and explains how tokens interact with other tokens.
    """,

    "page_016": """
    Solutions to problems faced by self-attention:
    use learnable parameters and different Query, Key,
    and Value vectors. Matrices WQ, WK and WV transform
    embeddings. Similarity is calculated using the
    dot product between Query and Key vectors.
    Similarities can be asymmetric.
    """,

    "page_017": """
    This page explains limitations of self-attention
    and illustrates relationships between input tokens.
    """,

    "page_018": """
    This page discusses Transformer architecture and
    attention mechanisms, with diagrams illustrating
    relationships between tokens.
    """
}

# Store descriptions
for page_id, description in descriptions.items():

    embedding = embedder.embed(description)

    store.add_description(
        page_id=page_id,
        embedding=embedding,
        description=description
    )

# Search
query_embedding = embedder.embed(question)

results = store.search(
    query_embedding,
    top_k=4
)

print("\nDESCRIPTION RAG RESULTS\n")

for i, page_id in enumerate(results["ids"][0]):

    print(
        f"Page: {results['metadatas'][0][i]['page_number']}"
    )

    print(
        f"Distance: {results['distances'][0][i]:.4f}"
    )

    print()