from multimodal_embedding import MultimodalEmbedder


embedder = MultimodalEmbedder()

image_embedding = embedder.embed_image(
    "C:\\Users\\Tanu Bhardwaj\\Downloads\\Intelligent_QA_V3\\backend\\page_images\\page_021.png"
)

text_embedding = embedder.embed_text(
    "What are Query, Key and Value?"
)

print("Image embedding shape:", image_embedding.shape)
print("Text embedding shape:", text_embedding.shape)