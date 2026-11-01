import chromadb


class TextVectorStore:

    def __init__(
        self,
        db_path="./chroma_db",
        collection_name="page_text"
    ):
        self.client = chromadb.PersistentClient(
            path=db_path
        )

        self.collection_name = collection_name

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            configuration={
                "hnsw": {
                    "space": "cosine"
                }
            }
        )

    def reset(self):

        self.client.delete_collection(
            self.collection_name
        )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            configuration={
                "hnsw": {
                    "space": "cosine"
                }
            }
        )

    def add_page(self, page_id, embedding, metadata):

        self.collection.upsert(
            ids=[page_id],
            embeddings=[embedding.tolist()],
            metadatas=[metadata]
        )

    
    def search(self, query_embedding, top_k=5, max_distance=0.70):

    # Reacquire the current collection after any reset
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
        configuration={"hnsw": {"space": "cosine"}}
    )

        count = self.collection.count()

        if count == 0:
            return {
            "ids": [[]],
            "distances": [[]],
            "metadatas": [[]]
        }

        results = self.collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(top_k, count)
    )

        filtered = {
        "ids": [[]],
        "distances": [[]],
        "metadatas": [[]]
    }

        for i, distance in enumerate(results["distances"][0]):
            if distance <= max_distance:
                filtered["ids"][0].append(results["ids"][0][i])
                filtered["distances"][0].append(distance)
                filtered["metadatas"][0].append(
                results["metadatas"][0][i]
                )

        return filtered
    
    def get_page_text(self, page_number):
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            configuration={"hnsw": {"space": "cosine"}}
        )

        result = self.collection.get(
            ids=[f"page_{page_number:03d}"],
            include=["metadatas"]
        )

        if not result["metadatas"]:
            return None

        return result["metadatas"][0].get("text")
