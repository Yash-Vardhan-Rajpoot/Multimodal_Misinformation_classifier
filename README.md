# 🧠 Multimodal Misinformation Classifier

A **Multimodal AI system** that detects potential misinformation by analyzing the **semantic similarity between an image and its caption** using the CLIP vision-language model.

The system compares **image embeddings** and **text embeddings** to determine whether the caption accurately represents the image.

---

## 🚀 Live Demo

🔗 **Try the app here:**
https://multimodalmisinformationclassifier.streamlit.app/

Upload an image and provide a caption to check whether the caption aligns with the visual content.

The application is deployed using **Streamlit**.

---

## 💡 Project Idea

Misinformation frequently spreads through **misleading captions paired with images**.

Example:

Image → Flood in Brazil
Caption → *"Massive flood in India today"*

Even though the image is real, the caption is **misleading**.

This project detects such cases using **multimodal similarity analysis**.

---

## 🏗️ System Architecture

Image → CLIP Encoder → Image Embedding
Caption → CLIP Encoder → Text Embedding

↓

Cosine Similarity

↓

Threshold Decision

↓

Classification

* ✅ Likely Real Information
* ⚠️ Potential Misinformation

The system uses **CLIP (Contrastive Language-Image Pretraining)** to encode both image and text into the **same embedding space**.

---

## 🧠 Model Used

Pretrained Vision-Language Model:

```
openai/clip-vit-base-patch32
```

CLIP learns a shared embedding space that allows comparison between **visual content and textual descriptions**.

---

## 📂 Project Structure

```
Multimodal_Misinformation_classifier
│
├── Multimodal Misinformation Classifier
│   │
│   ├── App
│   │   └── app_fixed.py          # Streamlit UI
│   │
│   ├── Model
│   │   └── clip_model.py         # CLIP encoder
│   │
│   └── Utils
│       └── similarity.py         # cosine similarity computation
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```
git clone https://github.com/Yash-Vardhan-Rajpoot/Multimodal_Misinformation_classifier.git
cd Multimodal_Misinformation_classifier
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```
streamlit run "Multimodal Misinformation Classifier/App/app_fixed.py"
```

The application will start locally and open in your browser.

---

## 🖥️ How to Use

1. Upload an image
2. Enter a caption describing the image
3. The model computes similarity between image and caption
4. Based on the similarity score, the system predicts:

* **Likely Real Information**
* **Potential Misinformation**

---

## 📊 Similarity Threshold

Default threshold:

```
0.26
```

If similarity score < threshold → **Potential Misinformation**

---

## 🛠️ Tech Stack

* Python
* PyTorch
* Transformers
* CLIP Model
* Streamlit
* NumPy
* Pillow

---

## 🌐 Deployment

The project is deployed using **Streamlit** and accessible here:

https://multimodalmisinformationclassifier.streamlit.app/

---

## 📌 Future Improvements

* Train a **custom multimodal classifier**
* Integrate **news fact-checking datasets**
* Add **image reverse search**
* Improve detection using **transformer-based multimodal fusion**
* Build **dataset for misinformation detection**

---

## 👨‍💻 Author

**Yash Vardhan Rajpoot**
NIT Patna

GitHub:
https://github.com/Yash-Vardhan-Rajpoot

---

## ⭐ Support

If you found this project useful, consider **starring the repository** on GitHub.
