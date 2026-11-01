from pathlib import Path

from multimodal_embedding import MultimodalEmbedder
from vector_store import VectorStore
from vlm import VLM


BASE_DIR = Path(__file__).resolve().parent.parent


embedder = MultimodalEmbedder()

vector_store = VectorStore(
    db_path=str(BASE_DIR / "chroma_db")
)


question = "What are Query, Key and Value?"


# --------------------------------
# 1. Convert question to embedding
# --------------------------------

query_embedding = embedder.embed_text(question)


# --------------------------------
# 2. Retrieve relevant pages
# --------------------------------

results = vector_store.search(
    query_embedding,
    top_k=1
)


image_paths = []
page_numbers = []


for i in range(len(results["ids"][0])):

    metadata = results["metadatas"][0][i]

    image_paths.append(
        metadata["image_path"]
    )

    page_numbers.append(
        metadata["page_number"]
    )


print("\nRetrieved pages:")

for page in page_numbers:
    print(f"Page {page}")


# --------------------------------
# 3. Send retrieved pages to VLM
# --------------------------------

vlm = VLM()

answer = vlm.answer(
    question,
    image_paths,
    page_numbers
)


print("\n==============================")
print("ANSWER")
print("==============================")

print(answer)