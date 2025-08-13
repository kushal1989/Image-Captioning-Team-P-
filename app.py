import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="📷 Image Caption Generator", page_icon="📷", layout="centered")

# -------------------------
# Load models & tokenizer
# -------------------------
@st.cache_resource
def load_mobilenet():
    base_model = MobileNetV2(weights="imagenet")
    return Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)

@st.cache_resource
def load_caption_model():
    return tf.keras.models.load_model("mymodel.h5")

@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        return pickle.load(f)

mobilenet_model = load_mobilenet()
caption_model = load_caption_model()
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
