import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
import gdown
import os
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image
from tensorflow.keras.layers import LSTM, Bidirectional, Embedding, Dense, Dropout

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="📷 Image Caption Generator", page_icon="📷", layout="centered")

# -------------------------
# Download model from Google Drive
# -------------------------
MODEL_PATH = "mymodel.h5"
TOKENIZER_PATH = "tokenizer.pkl"  # Keep this in GitHub or also use Drive

@st.cache_resource


@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_PATH):
        file_id = "1tBVQvprUw6woPkhNF2d-ZFhdywiSsLwY"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "LSTM": LSTM,
            "Bidirectional": Bidirectional,
            "Embedding": Embedding,
            "Dense": Dense,
            "Dropout": Dropout
        }
    )


@st.cache_resource
def load_tokenizer():
    with open(TOKENIZER_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_mobilenet():
    base_model = MobileNetV2(weights="imagenet")
    return Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)

mobilenet_model = load_mobilenet()
caption_model = download_model()
tokenizer = load_tokenizer()

# -------------------------
# Caption prediction
# -------------------------
def get_word_from_index(index, tokenizer):
    for word, idx in tokenizer.word_index.items():
        if idx == index:
            return word
    return None

def predict_caption(image_features, max_length=34):
    caption = "startseq"
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([caption])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = caption_model.predict([image_features, sequence], verbose=0)
        predicted_index = np.argmax(yhat)
        predicted_word = get_word_from_index(predicted_index, tokenizer)
        if predicted_word is None:
            break
        caption += " " + predicted_word
        if predicted_word == "endseq":
            break
    return caption.replace("startseq", "").replace("endseq", "").strip()

# -------------------------
# UI
# -------------------------
st.title("📷 Image Caption Generator")
st.write("Upload an image and get an AI-generated caption.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Generating caption..."):
        img = image.resize((224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Extract features
        features = mobilenet_model.predict(img_array, verbose=0)

        # Predict caption
        caption = predict_caption(features)

    st.success("Caption generated!")
    st.markdown(f"**Caption:** {caption}")
