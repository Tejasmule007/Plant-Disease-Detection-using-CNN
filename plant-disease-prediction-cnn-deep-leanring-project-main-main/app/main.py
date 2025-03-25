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
    # ✅ Set working directory and model paths
    working_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(working_dir, "app", "trained_model")
    model_h5_path = os.path.join(model_dir, "plant_disease_prediction_model.h5")
    model_keras_path = os.path.join(model_dir, "plant_disease_prediction_model.keras")

    # ✅ Google Drive Model Download URL
    gdrive_file_id = "1WLJk_JlWYL-1M8enmRgiCx3ddYNJwDUv"
    gdrive_url = f"https://drive.google.com/uc?id={gdrive_file_id}"

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model = None
    if os.path.exists(model_keras_path):
        model = tf.keras.models.load_model(model_keras_path)

    with open(f"{working_dir}/class_indices.json", "r") as f:
        class_indices = json.load(f)
    class_indices = {int(k): v for k, v in class_indices.items()}

    def load_and_preprocess_image(image, target_size=(224, 224)):
        img = image.resize(target_size)
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array.astype('float32') / 255.0
        return img_array

    def predict_image_class(model, image, class_indices):
        if model is None:
            return "No Model", 0.0

        preprocessed_img = load_and_preprocess_image(image)
        predictions = model.predict(preprocessed_img)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_name = class_indices.get(predicted_class_index, "Unknown Disease")
        confidence_score = np.max(predictions) * 100
        return predicted_class_name, confidence_score

    def get_disease_description(disease_name):
        prompt = f"""Provide a detailed description of the plant disease '{disease_name}'. Include:
        1. A brief description of the disease.
        2. Possible causes and environmental conditions leading to it.
        3. Treatment and preventive measures.
        4. Associated plant species that may also be affected.
        """
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

    def chatbot_response(user_query):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a plant expert. Answer user queries about plant diseases and plant care."},
                    {"role": "user", "content": user_query}
                ],
                max_tokens=500,
                temperature=0.7
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            return f"❌ Error: {e}"

    # ✅ Create Tabs
    tab1, tab2 = st.tabs(["🖼 Disease Detection", "💬 Chat with AI"])

    # 🖼 Tab 1: Disease Detection
    with tab1:
        st.subheader("📸 Capture or Upload Images for Disease Detection")

        camera_image = st.camera_input("📷 Take a photo of a plant leaf")
        uploaded_images = st.file_uploader("📤 Upload plant images (Multiple Allowed)", type=["jpg", "jpeg", "png"],
                                           accept_multiple_files=True)

        images_to_process = []

        if camera_image:
            images_to_process.append(Image.open(camera_image))
        if uploaded_images:
            for img_file in uploaded_images:
                images_to_process.append(Image.open(img_file))

        if images_to_process:
            if st.button("🔍 Submit for Analysis"):
                st.subheader("🖼 Processed Images & Results")
                for idx, image in enumerate(images_to_process):
                    col1, col2 = st.columns([1, 2])

                    with col1:
                        resized_img = image.resize((200, 200))
                        st.image(resized_img, caption=f"Image {idx + 1}", use_column_width=True)

                    with col2:
                        predicted_disease, confidence = predict_image_class(model, image, class_indices)
                        disease_details = get_disease_description(predicted_disease)

                        st.success(f"🌱 **Prediction:** {predicted_disease}")
                        st.info(f"🔢 **Confidence:** {confidence:.2f}%")
                        st.markdown(f"📖 **Disease Information:**\n\n{disease_details}")
            else:
                st.info("👉 Click 'Submit for Analysis' to see results.")
        else:
            st.info("📥 Please upload or capture at least one image to begin.")

    # 💬 Tab 2: Chatbot
    with tab2:
        st.subheader("💬 Ask the Plant AI Expert")
        user_input = st.text_input("Type your question here (e.g., How to treat black rot on apples?)", "")

        if st.button("💡 Get Answer"):
            if user_input:
                response = chatbot_response(user_input)
                st.write(f"🤖 **AI Response:**\n\n{response}")
            else:
                st.warning("Please enter a question before submitting.")
