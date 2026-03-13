import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


class CLIPEncoder:

    def __init__(self, model_name="openai/clip-vit-base-patch32"):

        # Select device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load CLIP model
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)

        # Load processor
        self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)

    def encode_image(self, image_input):

        # Handle different input types

        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")

        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")

        else:
            image = Image.open(image_input).convert("RGB")

        # Preprocess image
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        # Get image features
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)

        return image_features

    def encode_text(self, text):

        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)

        return text_features