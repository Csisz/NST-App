import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import io

st.title("🎨 Neural Style Transfer App")

# Upload images
content_img = st.file_uploader("Upload your content image", type=["jpg", "png"])
style_img = st.file_uploader("Upload your style image", type=["jpg", "png"])

def load_and_process_image(img_file):
    img = Image.open(img_file).convert("RGB").resize((256, 256))
    img = np.array(img) / 255.0
    return tf.constant(img[np.newaxis, ...], dtype=tf.float32)

if content_img and style_img:
    content = load_and_process_image(content_img)
    style = load_and_process_image(style_img)

    # Load model
    with st.spinner("Applying style..."):
        model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
        stylized_image = model(content, style)[0]

    # Convert and display
    output_img = tf.clip_by_value(stylized_image[0] * 255.0, 0, 255).numpy().astype(np.uint8)
    st.image(output_img, caption="Stylized Output", use_column_width=True)
