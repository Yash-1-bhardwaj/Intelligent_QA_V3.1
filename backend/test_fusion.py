from pathlib import Path

from text_embedding import TextEmbedder
from text_vector_store import TextVectorStore

from multimodal_embedding import MultimodalEmbedder
from vector_store import VectorStore

from fusion import Fusion


BASE_DIR = Path(__file__).resolve().parent.parent

question = "what are solutions of problem faced by self-attention transformer?"


# -------------------------
# TEXT RAG
# -------------------------

text_embedder = TextEmbedder()

text_store = TextVectorStore(
    db_path=str(BASE_DIR / "chroma_db")
)

text_embedding = text_embedder.embed(question)

text_results = text_store.search(
    text_embedding,
    top_k=5
)


# -------------------------
# IMAGE RAG
# -------------------------

image_embedder = MultimodalEmbedder()

image_store = VectorStore(
    db_path=str(BASE_DIR / "chroma_db")
)

image_embedding = image_embedder.embed_text(question)

image_results = image_store.search(
    image_embedding,
    top_k=5
)


# -------------------------
# FUSION
# -------------------------

fusion = Fusion()

ranked_pages = fusion.combine(
    text_results,
    image_results
)


print("\nFUSED RESULTS\n")

for rank, (page_id, score) in enumerate(
    ranked_pages,
    start=1
):

    print(
        f"{rank}. {page_id} | score: {score:.3f}"
    )