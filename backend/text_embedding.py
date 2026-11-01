from sentence_transformers import SentenceTransformer


class TextEmbedder:

    def __init__(self):
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    def embed(self, text):
        return self.model.encode(text)


if __name__ == "__main__":

    embedder = TextEmbedder()

    text = "What are Query, Key and Value?"

    embedding = embedder.embed(text)

    print("Embedding shape:", embedding.shape)