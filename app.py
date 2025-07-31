import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os

st.title("Neural Style Transfer")

# Upload image
uploaded_image = st.file_uploader("Upload a content image", type=["jpg", "png", "jpeg"])
if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Original Image", use_column_width=True)

    # Preprocess image
    img_array = np.array(image.resize((256, 256))).astype(np.float32) / 255.0
    content = tf.convert_to_tensor(img_array)[tf.newaxis, ...]

    # Load your model
    model = tf.keras.models.load_model("style_transfer_full_model.h5", custom_objects={"InstanceNormalization": InstanceNormalization})

    # Run inference
    stylized = model(content)[0].numpy()
    stylized = np.clip(stylized * 255, 0, 255).astype(np.uint8)

    st.image(stylized, caption="Stylized Image", use_column_width=True)
