import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


MODEL_NAME = "openai/clip-vit-base-patch32"


class MultimodalEmbedder:

    def __init__(self):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("Using device:", self.device)

        self.model = CLIPModel.from_pretrained(MODEL_NAME)
        self.processor = CLIPProcessor.from_pretrained(MODEL_NAME)

        self.model.to(self.device)
        self.model.eval()


    def embed_image(self, image_path):

        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(
        images=image,
        return_tensors="pt"
    )

        inputs = {
        key: value.to(self.device)
        for key, value in inputs.items()
    }

        with torch.no_grad():

            embedding = self.model.get_image_features(**inputs)

            if not isinstance(embedding, torch.Tensor):
                embedding = embedding.pooler_output

        embedding = embedding / embedding.norm(
        dim=-1,
        keepdim=True
    )

        return embedding[0].cpu().numpy()


    def embed_text(self, text):

        inputs = self.processor(
        text=[text],
        return_tensors="pt",
        padding=True
    )

        inputs = {
        key: value.to(self.device)
        for key, value in inputs.items()
    }

        with torch.no_grad():

            embedding = self.model.get_text_features(**inputs)

            if not isinstance(embedding, torch.Tensor):
                embedding = embedding.pooler_output

        embedding = embedding / embedding.norm(
        dim=-1,
        keepdim=True
    )

        return embedding[0].cpu().numpy()