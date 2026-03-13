import streamlit as st
from PIL import Image
from Model.clip_model import CLIPEncoder
from Utils.similarity import compute_similarity
import torch

st.title("Multimodal Misinformation Detector")

st.write("Upload an image and provide a caption. The system will check if they match.")

# Load model
@st.cache_resource
def load_model():
    return CLIPEncoder()

model = load_model()

# Image uploader
uploaded_image = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# Caption input
caption = st.text_input("Enter Caption")

if uploaded_image and caption:

    image = Image.open(uploaded_image).convert("RGB")

    st.image(image, caption="Uploaded Image", width="stretch")

    if st.button("Check Post"):

        with torch.no_grad():
            # image already PIL.Image
            image_embedding = model.encode_image(image)
            text_embedding = model.encode_text(caption)

            score = compute_similarity(image_embedding, text_embedding)

        st.write("### Similarity Score:", round(score, 3))

        if score > 0.26:
            st.success("Caption matches the image")
        else:
            st.error("Potential Misinformation Detected")
