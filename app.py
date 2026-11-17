
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

st.set_page_config(page_title="Potato CNN", page_icon="🥔")

ROOT = Path(__file__).resolve().parent
IMG_SIZE = (128, 128)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(ROOT / "potato_cnn_model.keras")

@st.cache_data
def load_classes():
    return (ROOT / "classification_report.txt").read_text().splitlines()


# Home Page
st.title("🥔 Potato Leaf Classification")
st.write("Upload a potato leaf image to predict its disease.")

st.divider()

uploaded_file = st.file_uploader(
    "Upload Potato Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Leaf Image", width=300)

    if st.button("Predict"):

        try:
            model = load_model()
            classes = load_classes()

            resized_image = image.resize(IMG_SIZE)

            x = np.expand_dims(
                np.array(resized_image, dtype=np.float32),
                axis=0
            )

            scores = model.predict(x, verbose=0)[0]
            best = np.argmax(scores)

            predicted_class = classes[best].replace("potato__", "")

            st.success("Prediction Completed!")

            st.subheader("Prediction Result")
            st.write(f"**Class:** {predicted_class}")
            st.write(f"**Confidence:** {scores[best]*100:.2f}%")

            st.subheader("Class Probabilities")

            for name, score in zip(classes, scores):
                st.write(f"{name.replace('potato__', '')}: {score*100:.2f}%")
                st.progress(float(score))

        except Exception as e:
            st.error(f"Error: {e}")


# About Section
# st.divider()

# st.header("About Project")

# st.write("Potato Leaf Classification using CNN.")

# st.write("**Technologies Used:**")
# st.write("- Python")
# st.write("- TensorFlow / Keras")
# st.write("- Streamlit")
# st.write("- NumPy")

# st.write("**Image Size:** 128 × 128")
# st.write("**Output Classes:** 3")
