import chromadb


class VectorStore:

    def __init__(
        self,
        db_path="./chroma_db",
        collection_name="page_images"
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


    
    def search(self, query_embedding, top_k=5, max_distance=0.75):

    # Refresh the collection reference before querying
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
                filtered["metadatas"][0].append(results["metadatas"][0][i])

        return filtered
