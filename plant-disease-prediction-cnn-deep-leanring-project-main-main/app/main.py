import os
import json
import gdown
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
import openai
import h5py

# Set page config and inject background + UI styles
st.set_page_config(page_title="Plant Diagnosis & AI Bot 🌿", layout="wide")
st.markdown("""
    <style>
    .main {
        background: linear-gradient(to bottom right, #e3fceb, #f6fff9),
                    url("https://www.transparenttextures.com/patterns/leaf.png");
        background-size: cover;
        background-attachment: fixed;
    }

    .stApp {
        font-family: 'Segoe UI', sans-serif;
    }

    /* Layout styling */
    div[data-testid="column"] > div {
        padding: 0.5rem;
        min-height: 320px;
    }

    section[data-testid="stFileUploader"] {
        padding-top: 1rem;
        padding-bottom: 1rem;
        background-color: #ffffffcc;
        border-radius: 10px;
        box-shadow: 0 0 5px rgba(0,0,0,0.1);
    }

    .element-container:has([data-testid="stCameraInput"]) {
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
        background-color: #ffffffcc;
        border-radius: 10px;
        box-shadow: 0 0 5px rgba(0,0,0,0.1);
    }

    .stButton > button {
        background-color: #4CAF50;
        color: white;
        padding: 0.4rem 1.2rem;
        border-radius: 6px;
        font-size: 15px;
        font-weight: bold;
    }

    .stButton > button:hover {
        background-color: #45a049;
    }

    .result-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# App Title
st.markdown("## 🌿 Plant Disease Detection & Chatbot Assistant")
st.markdown("A smart plant care tool powered by Deep Learning and GPT-4.")

# API Key input
if "api_key_set" not in st.session_state:
    st.session_state.api_key_set = False

if not st.session_state.api_key_set:
    with st.expander("🔐 Set OpenAI API Key"):
        api_key = st.text_input("Enter your OpenAI API Key:", type="password")
        if st.button("Submit"):
            if api_key:
                openai.api_key = api_key
                st.session_state.api_key_set = True
                st.success("✅ API Key set successfully!")
                if st.button("Next ➡️"):
                    st.experimental_rerun()
            else:
                st.warning("Please enter a valid key.")
else:
    # Paths
    working_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(working_dir, "app", "trained_model")
    model_h5_path = os.path.join(model_dir, "plant_disease_prediction_model.h5")
    gdrive_file_id = "11W9S_tSxzWsHmaje-Oc4upPJHiQCxyzK"

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Download and load model
    model = None
    if not os.path.exists(model_h5_path):
        with st.spinner("📦 Downloading model from Google Drive..."):
            try:
                gdown.download(id=gdrive_file_id, output=model_h5_path, quiet=False)
                if os.path.exists(model_h5_path):
                    st.success("✅ Model downloaded successfully!")
                else:
                    st.error("❌ Download failed: file not found after attempt.")
                    st.stop()
            except Exception as e:
                st.error(f"❌ Error during download: {e}")
                st.stop()

    try:
        model = tf.keras.models.load_model(model_h5_path)
        st.success("✅ Model loaded and ready!")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

    # Load class labels
    try:
        with open(os.path.join(working_dir, "class_indices.json"), "r") as f:
            class_indices = json.load(f)
        class_indices = {int(k): v for k, v in class_indices.items()}
    except Exception as e:
        st.error(f"❌ Error loading class labels: {e}")
        st.stop()

    # Helper functions
    def load_and_preprocess_image(image, target_size=(224, 224)):
        img = image.resize(target_size)
        img_array = np.expand_dims(np.array(img).astype("float32") / 255.0, axis=0)
        return img_array

    def predict_image_class(model, image, class_indices):
        processed_img = load_and_preprocess_image(image)
        predictions = model.predict(processed_img)
        predicted_idx = np.argmax(predictions, axis=1)[0]
        class_name = class_indices.get(predicted_idx, "Unknown Disease")
        confidence = np.max(predictions) * 100
        return class_name, confidence

    def get_disease_description(disease_name):
        prompt = f"""Provide a detailed description of the plant disease '{disease_name}'. Include:
        - Overview
        - Causes and conditions
        - Treatment and prevention
        - Plants that may be affected"""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert in plant diseases."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"❌ GPT Error: {e}"

    def chatbot_response(query):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You're a helpful plant health expert."},
                    {"role": "user", "content": query}
                ],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"❌ GPT Error: {e}"

    # Tabs
    tab1, tab2 = st.tabs(["🖼 Diagnose Plant Disease", "💬 Ask the AI Expert"])

    # Tab 1: Plant Disease Detection
    with tab1:
        st.markdown("### Upload or Capture Plant Leaf Images")

        col1, col2 = st.columns([1, 2])
        with col1:
            camera_image = st.camera_input("📷 Take a photo")
        with col2:
            uploaded_images = st.file_uploader("📤 Upload image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

        images = []
        if camera_image:
            images.append(Image.open(camera_image))
        if uploaded_images:
            for img in uploaded_images:
                images.append(Image.open(img))

        if images:
            if st.button("🔍 Analyze Image(s)"):
                for idx, image in enumerate(images):
                    pred_class, conf = predict_image_class(model, image, class_indices)
                    details = get_disease_description(pred_class)

                    with st.container():
                        st.markdown(f"<div class='result-card'>", unsafe_allow_html=True)
                        st.image(image.resize((200, 200)), caption=f"Leaf Image {idx + 1}", use_column_width=False)
                        st.markdown(f"**🪴 Prediction:** `{pred_class}`")
                        st.markdown(f"**🎯 Confidence:** `{conf:.2f}%`")
                        st.markdown(f"**📚 Disease Info:**\n\n{details}")
                        st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("👉 Click 'Analyze Image(s)' to start.")
        else:
            st.info("📥 Upload or capture an image to begin.")

    # Tab 2: GPT-4 Chatbot
    with tab2:
        st.markdown("### 💬 Ask the Plant AI Expert Anything")
        user_query = st.text_input("Ask something like: *How to treat leaf blight in tomatoes?*")
        if st.button("💡 Get Answer"):
            if user_query.strip():
                st.markdown("#### 🤖 AI Response")
                st.write(chatbot_response(user_query))
            else:
                st.warning("Please enter a question first.")
