import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from datetime import datetime
import csv
import os
import json
import warnings
import requests # <-- ADDED THIS IMPORT

# Commented out chatbot as we don't have the file
from chatbot_module import chatbot_interface 
warnings.filterwarnings("ignore")

# --- THIS FUNCTION IS NOW FIXED ---

# The URL to your model file
MODEL_URL = "https://huggingface.co/fekferijrrner4483g-d/leaf-disease-model/resolve/main/leaf_disease_model1.h5"

# The local path where we'll save the model
MODEL_PATH = "leaf_disease_model1.h5"

@st.cache_resource
def load_leaf_model():
    """
    Downloads the model from the URL if it doesn't exist locally,
    then loads it into memory. Caches the loaded model.
    """
    # 1. Download the model file if it doesn't exist locally
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...") # For your console
        with st.spinner("Downloading model... (this happens only once)"):
            try:
                r = requests.get(MODEL_URL)
                r.raise_for_status() # Check for download errors
                with open(MODEL_PATH, "wb") as f:
                    f.write(r.content)
                print("✅ Model downloaded.")
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Failed to download model: {e}")
                return None

    # 2. Load the model from the local file
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully from local file.")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model from disk: {str(e)}")
        return None

# --- END OF FIXED FUNCTION ---


# Predict disease
def predict_disease(img, model, class_names):
    x = image.img_to_array(img)
    x = x / 255.0  # Normalize
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x, verbose=0)
    predicted_index = np.argmax(preds)
    disease_name = class_names[predicted_index]
    accuracy = float(preds[0][predicted_index])
    return disease_name, accuracy

# Save results
def save_results_to_csv(results):
    file_name = "results.csv"
    with open(file_name, "a+", newline="") as f:
        writer = csv.writer(f)
        for result in results:
            writer.writerow([result[0], result[1], result[2], result[3]])
    return file_name

# Display previous results
def display_previous_results():
    try:
        df = pd.read_csv("results.csv", header=None, names=["Date", "Time", "Disease", "Model Accuracy"])
        st.write("Previous Results:")
        st.dataframe(df)
    except FileNotFoundError:
        st.warning("No previous results found.")


# Main
def main():
    st.title("🌿 AI For Bharat's Agriculture")

    # Sidebar with mode selection
    st.sidebar.title("Options")
    
    # Mode selection
    mode = st.sidebar.selectbox(
        "Choose Mode:",
        ["🔍 Image Prediction", "🤖 Chatbot"],
        help="Select the mode you want to use"
    )
    
    # Display previous results button
    if st.sidebar.button("📂 Display Previous Results"):
        display_previous_results()

    # Download results button
    if "results" in st.session_state:
        with open(st.session_state.results, "rb") as f:
            data = f.read()
        st.sidebar.download_button(
            "⬇️ Download Results CSV",
            data,
            file_name=os.path.basename(st.session_state.results),
            key="download_button",
        )

    # Main content based on selected mode
    if mode == "🔍 Image Prediction":
        st.subheader("📸 Tomato Leaf Disease Detection")
        uploaded_file = st.file_uploader("📤 Upload a leaf image...", type=["jpg", "jpeg", "png"])

        model = load_leaf_model() # This will now work
        
        if model is None:
            # The error message is already shown by load_leaf_model()
            st.warning("Model is not loaded. Please check the error message above.")
            return

        # Load class names from JSON
        try:
            with open("class_names.json", "r") as f:
                class_names = json.load(f)
        except Exception as e:
            st.error(f"❌ Error loading class names: {str(e)}")
            st.error("Please make sure a 'class_names.json' file exists.")
            return

        if uploaded_file is not None:
            img = image.load_img(uploaded_file, target_size=(224, 224))
            st.image(img, caption="Uploaded Leaf Image", use_container_width=True)

            if st.button("🔍 Predict Disease"):
                disease_name, accuracy = predict_disease(img, model, class_names)
                
                st.success(f"Disease Detected: **{disease_name}**")
                st.info(f"Model Accuracy: **{accuracy * 100:.2f}%**")
                
                now = datetime.now()
                results = [(now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), disease_name, accuracy)]
                file_name = save_results_to_csv(results)
                st.success(f"✅ Results saved to `{file_name}`")

                st.session_state.results = file_name
    
    elif mode == "🤖 Chatbot":
        # st.info("Chatbot interface is not enabled in this snippet.")
        chatbot_interface()


if __name__ == "__main__":
    main()
