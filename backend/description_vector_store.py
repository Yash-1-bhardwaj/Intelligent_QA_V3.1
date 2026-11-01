import chromadb


class DescriptionVectorStore:

    def __init__(
        self,
        db_path="./chroma_db",
        collection_name="page_descriptions"
    ):
        self.client = chromadb.PersistentClient(
            path=db_path
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            configuration={
                "hnsw": {
                    "space": "cosine"
                }
            }
        )

    def add_description(
        self,
        page_id,
        embedding,
        description
    ):
        self.collection.upsert(
            ids=[page_id],
            embeddings=[embedding.tolist()],
            documents=[description],
            metadatas=[{
                "page_number": int(page_id.split("_")[1])
            }]
        )

    def search(self, query_embedding, top_k=3):
        return self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )