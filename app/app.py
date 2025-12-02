import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Astronomical Object Classifier", layout="centered")

# =========================================================
# FIXED MODEL PATH
MODEL_PATH = "model/astronomical_classifier.keras"
CLASS_NAMES = [
    "constellation",
    "cosmos space",
    "galaxies",
    "nebula",
    "planets",
    "stars",
]
CONFIDENCE_THRESHOLD = 0.60
# =========================================================

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224), Image.Resampling.LANCZOS)
    arr = np.array(image) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict(image):
    processed = preprocess_image(image)
    preds = model.predict(processed)[0]
    top_idx = np.argmax(preds)
    top_conf = float(preds[top_idx])
    top_label = CLASS_NAMES[top_idx]

    # --- Unknown object rule ---
    if top_conf < CONFIDENCE_THRESHOLD:
        return "Unknown Object", top_conf
    return top_label, top_conf

# =========================================================
# UI
# =========================================================

st.title("🔭 Astronomical Object Classifier")
st.write("Upload an image to classify.")

uploaded_file = st.file_uploader("Choose an Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    try:
        label, confidence = predict(image)
        st.subheader("🔍 Prediction Results")
        st.write(f"**Prediction:** {label}")
        st.write(f"**Confidence:** {confidence:.2f}")

        if label == "Unknown Object":
            st.warning("This image does not match any astronomical object from the dataset.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
