from pathlib import Path
import numpy as np

from multimodal_embedding import MultimodalEmbedder
from text_embedding import TextEmbedder

from vector_store import VectorStore
from text_vector_store import TextVectorStore

from fusion import Fusion
from vlm import VLM
import time

class QAPipeline:

    def __init__(self, base_dir):

        self.base_dir = Path(base_dir)

        # Embedding models
        self.image_embedder = MultimodalEmbedder()
        self.text_embedder = TextEmbedder()

        # Vector stores
        self.image_store = VectorStore(
            db_path=str(self.base_dir / "chroma_db")
        )

        self.text_store = TextVectorStore(
            db_path=str(self.base_dir / "chroma_db")
        )

        # Fusion and VLM
        self.fusion = Fusion()
        self.vlm = VLM()

        # Page image directory
        self.page_dir = (
            self.base_dir / "backend" / "page_images"
        )

    def get_page_image(self, page_id, image_results):

        # First, try to get the image path from image retrieval
        image_ids = image_results.get("ids", [[]])[0]
        metadatas = image_results.get("metadatas", [[]])[0]

        for i, result_id in enumerate(image_ids):
            if result_id == page_id:
                metadata = metadatas[i]

                image_path = Path(metadata["image_path"])

                if image_path.exists():
                    return image_path

        # Fallback: construct the path from the page ID
        image_path = self.page_dir / f"{page_id}.png"

        if image_path.exists():
            return image_path

        return None

    @staticmethod
    def cosine_similarity(a, b):

        a = np.asarray(a).flatten()
        b = np.asarray(b).flatten()

        denominator = np.linalg.norm(a) * np.linalg.norm(b)

        if denominator == 0:
            return -1.0

        return float(np.dot(a, b) / denominator)

    def ask(self, question):

        # -------------------------
        # 1. TEXT RETRIEVAL
        # -------------------------
        start = time.perf_counter()
        text_embedding = self.text_embedder.embed(question)

        text_results = self.text_store.search(
            text_embedding,
            top_k=5
        )
        print(f"Text retrieval: {time.perf_counter() - start:.2f}s")

        # -------------------------
        # 2. IMAGE RETRIEVAL
        # -------------------------
        start = time.perf_counter()

        image_embedding = self.image_embedder.embed_text(
            question
        )

        image_results = self.image_store.search(
            image_embedding,
            top_k=5
        )
        print(f"Image retrieval: {time.perf_counter() - start:.2f}s")

        # -------------------------
        # 3. FUSION
        # -------------------------
        start = time.perf_counter()
        ranked_pages = self.fusion.combine(
            text_results,
            image_results
        )
        print(f"Fusion: {time.perf_counter() - start:.2f}s")
        # Consider up to five fused candidates
        start = time.perf_counter()
        candidate_pages = ranked_pages[:3]
        print("\n--- TEXT RETRIEVAL ---")
        for page_id, distance in zip(
    text_results["ids"][0],
    text_results["distances"][0]
):
            print(page_id, f"distance={distance:.4f}")

        print("\n--- IMAGE RETRIEVAL ---")
        for page_id, distance in zip(
    image_results["ids"][0],
    image_results["distances"][0]
):
            print(page_id, f"distance={distance:.4f}")

        print("\n--- FUSED RANKING ---")
        for page_id, score in ranked_pages:
            print(page_id, f"score={score:.4f}")
        print(f"Description-based reranking: {time.perf_counter() - start:.2f}s")
        # -------------------------
        # 4. DESCRIPTION-BASED RERANKING
        # -------------------------
        
        
        # -------------------------
        # 4. SELECT TOP RETRIEVED PAGE
        # -------------------------
        start = time.perf_counter()

        selected_image = None
        selected_page = None

        # Use the fused ranking as the reliable fallback.
        # No additional VLM calls are needed for page selection.
        for page_id, fusion_score in candidate_pages:
            image_path = self.get_page_image(
                page_id,
                image_results
            )

            if image_path is not None:
                selected_image = image_path
                selected_page = int(page_id.split("_")[1])
                break

        if selected_image is None:
            return {
                "question": question,
                "answer": "I couldn't find a usable page to answer this question.",
                "pages": []
            }

        print(
            f"Selected page: {selected_page} | "
            f"Selection time: {time.perf_counter() - start:.2f}s"
        )

        # -------------------------
        # 5. GENERATE ANSWER
        # -------------------------
        page_text = self.text_store.get_page_text(selected_page)

        if page_text and page_text.strip():
            answer = self.vlm.answer_from_text(
        question,
        [page_text],
        [selected_page]
    )
        else:
    # Fall back to the existing vision-language answer method
            answer = self.vlm.answer(
        question,
        [str(selected_image)],
        [selected_page]
    )

        return {
            "question": question,
            "answer": answer,
            "pages": [selected_page]
        }



        # # -------------------------
        # # 6. FINAL ANSWER FROM ORIGINAL IMAGE
        # # -------------------------

        # answer = self.vlm.answer(
        #     question,
        #     [str(selected_image)],
        #     [selected_page]
        # )

        # return {
        #     "question": question,
        #     "answer": answer,
        #     "pages": [selected_page]
        # }
