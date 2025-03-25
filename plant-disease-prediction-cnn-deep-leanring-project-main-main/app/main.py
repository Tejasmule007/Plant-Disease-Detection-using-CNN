import os
import json
import gdown
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
import openai
import h5py

# Streamlit App Title
st.title("🌿 Plant Disease Detection & Chatbot Assistant")

# First Page: Ask for API key and submit button
if "api_key_set" not in st.session_state:
    st.session_state.api_key_set = False

if not st.session_state.api_key_set:
    api_key = st.text_input("🔑 Enter your OpenAI API Key:", type="password")
    submit_button = st.button("Submit")

    if submit_button and api_key:
        openai.api_key = api_key
        st.session_state.api_key_set = True
        st.success("✅ API Key has been successfully set up!")
        if st.button("Next"):
            st.experimental_rerun()
else:
    # ✅ Paths
    working_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(working_dir, "app", "trained_model")
    model_h5_path = os.path.join(model_dir, "plant_disease_prediction_model.h5")

    # ✅ Google Drive File ID (from provided link)
    gdrive_file_id = "11W9S_tSxzWsHmaje-Oc4upPJHiQCxyzK"

    # ✅ Ensure model directory exists
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # ✅ Download model if not present
    model = None
    if not os.path.exists(model_h5_path):
        with st.spinner("📦 Downloading model from Google Drive..."):
            try:
                gdown.download(id=gdrive_file_id, output=model_h5_path, quiet=False)
                if os.path.exists(model_h5_path):
                    st.success("✅ Model downloaded successfully!")
                else:
                    st.error("❌ Download failed: File does not exist after attempt.")
                    st.stop()
            except Exception as e:
                st.error(f"❌ Download error: {e}")
                st.stop()

    # ✅ Load model
    try:
        model = tf.keras.models.load_model(model_h5_path)
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

    # ✅ Load class labels
    class_file_path = os.path.join(working_dir, "class_indices.json")
    try:
        with open(class_file_path, "r") as f:
            class_indices = json.load(f)
        class_indices = {int(k): v for k, v in class_indices.items()}
    except Exception as e:
        st.error(f"❌ Failed to load class index file: {e}")
        st.stop()

    # ✅ Image Preprocessing
    def load_and_preprocess_image(image, target_size=(224, 224)):
        img = image.resize(target_size)
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array.astype('float32') / 255.0
        return img_array

    # ✅ Image Prediction
    def predict_image_class(model, image, class_indices):
        preprocessed_img = load_and_preprocess_image(image)
        predictions = model.predict(preprocessed_img)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_name = class_indices.get(predicted_class_index, "Unknown Disease")
        confidence_score = np.max(predictions) * 100
        return predicted_class_name, confidence_score

    # ✅ GPT-4 Plant Disease Description
    def get_disease_description(disease_name):
        prompt = f"""Provide a detailed description of the plant disease '{disease_name}'. Include:
        1. A brief description of the disease.
        2. Causes and conditions.
        3. Treatment and prevention.
        4. Other plants affected."""
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
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            return f"❌ Error: {e}"

    # ✅ GPT-4 Chatbot
    def chatbot_response(user_query):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a plant expert. Answer questions about plant care and diseases."},
                    {"role": "user", "content": user_query}
                ],
                max_tokens=500,
                temperature=0.7
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            return f"❌ Error: {e}"

    # ✅ Tabs
    tab1, tab2 = st.tabs(["🖼 Disease Detection", "💬 Chat with AI"])

    # Tab 1: Disease Detection
    with tab1:
        st.subheader("📸 Upload or Capture Leaf Images")

        camera_image = st.camera_input("📷 Take a photo")
        uploaded_images = st.file_uploader("📤 Upload image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

        images_to_process = []
        if camera_image:
            images_to_process.append(Image.open(camera_image))
        if uploaded_images:
            for img_file in uploaded_images:
                images_to_process.append(Image.open(img_file))

        if images_to_process:
            if st.button("🔍 Submit for Analysis"):
                st.subheader("🖼 Results")
                for idx, image in enumerate(images_to_process):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        resized_img = image.resize((200, 200))
                        st.image(resized_img, caption=f"Image {idx+1}", use_column_width=True)
                    with col2:
                        predicted_disease, confidence = predict_image_class(model, image, class_indices)
                        disease_details = get_disease_description(predicted_disease)
                        st.success(f"🌱 **Prediction:** {predicted_disease}")
                        st.info(f"🔢 **Confidence:** {confidence:.2f}%")
                        st.markdown(f"📖 **Details:**\n\n{disease_details}")
            else:
                st.info("👉 Click 'Submit for Analysis' to view predictions.")
        else:
            st.info("📥 Please capture or upload at least one image.")

    # Tab 2: AI Chat
    with tab2:
        st.subheader("💬 Ask the Plant Expert")
        user_input = st.text_input("Type your question here...")
        if st.button("💡 Get Answer"):
            if user_input:
                answer = chatbot_response(user_input)
                st.write(f"🤖 **AI Response:**\n\n{answer}")
            else:
                st.warning("Please enter a question.")
