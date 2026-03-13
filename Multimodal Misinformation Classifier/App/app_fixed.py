import sys
import os
import streamlit as st
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Model.clip_model import CLIPEncoder
from Utils.similarity import compute_similarity


# Page title
st.title("Multimodal Misinformation Detector")

st.write("Upload an image and provide a caption to check if they match.")


# Load model only once
@st.cache_resource
def load_model():
    return CLIPEncoder()


model = load_model()


# Upload image
uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# Caption input
caption = st.text_input("Enter Caption")


# Similarity threshold
threshold = 0.26


# Display uploaded image
if uploaded_image is not None:

    uploaded_image.seek(0)
    image = Image.open(uploaded_image).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)


# Button to run detection
if st.button("Check Post"):

    if uploaded_image is None or caption.strip() == "":
        st.warning("Please upload an image and enter a caption.")

    else:

        # Reset pointer before reading again
        uploaded_image.seek(0)

        # Generate embeddings
        image_embedding = model.encode_image(uploaded_image)
        text_embedding = model.encode_text(caption)

        # Compute similarity
        score = compute_similarity(image_embedding, text_embedding)

        score = float(score)

        # Display similarity score
        st.subheader("Similarity Score")
        st.write(round(score, 3))

        st.write("Threshold:", threshold)

        # Classification
        if score >= threshold:
            st.success("Caption matches the image")
        else:
            st.error("Potential Misinformation Detected")
