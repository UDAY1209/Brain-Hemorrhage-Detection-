import tensorflow as tf
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import matplotlib.pyplot as plt  # Importing matplotlib for graph visualization

# Set page configuration
st.set_page_config(
    page_title="Brain Hemorrhage Detection",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load the model(s)
model_options = ['final.h5']  # Add more models if available
selected_model = st.sidebar.selectbox("Select Model", model_options)

try:
    model = tf.keras.models.load_model(selected_model)
    model_summary = f"Model loaded successfully: {model.name}"
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Sidebar for navigation and instructions
st.sidebar.title("Brain Hemorrhage Detection")
st.sidebar.write("Upload a brain scan image to detect hemorrhage.")
st.sidebar.write("Supported formats: JPG, PNG, JPEG")
st.sidebar.write(model_summary)

# Image preprocessing options
st.sidebar.subheader("Image Preprocessing")
convert_to_grayscale = st.sidebar.checkbox("Convert to Grayscale")
enhance_image = st.sidebar.checkbox("Enhance Image")

# Title
st.title('Brain Hemorrhage Detection')

# File uploader
fil = st.file_uploader('Upload an image', type=['jpg', 'png', 'jpeg'])

# Prediction history
if "history" not in st.session_state:
    st.session_state["history"] = []

def process_image(file):
    try:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        # Apply preprocessing options
        if convert_to_grayscale:
            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_GRAY2RGB)  # Convert back to 3 channels
        if enhance_image:
            opencv_image = cv2.equalizeHist(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY))
            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_GRAY2RGB)

        st.image(opencv_image, channels="RGB", caption="Uploaded Image (Processed)")

        # Convert image for TensorFlow processing
        file.seek(0)  # Reset file pointer
        image = Image.open(file)
        image = image.resize((130, 130))
        img = np.array(image)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0

        return img
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

def predict(img):
    with st.spinner('Processing...'):
        ans = model.predict(img)
        return ans

def save_results_to_file(results):
    output = ""
    for idx, res in enumerate(results):
        output += f"Prediction {idx + 1}: {res}\n"
    return output

def plot_confidence_graph(confidence):
    """Plot a bar chart for confidence levels."""
    labels = ['No Hemorrhage', 'Hemorrhage Detected']
    values = [100 - confidence, confidence]

    fig, ax = plt.subplots()
    ax.bar(labels, values, color=['blue', 'red'])
    ax.set_ylabel('Confidence (%)')
    ax.set_title('Hemorrhage Detection Confidence Levels')
    st.pyplot(fig)

if fil is not None:
    st.image(fil, caption="Uploaded Image", use_column_width=True)  # Display the uploaded image
    if st.button("Submit"):  # Add a Submit button
        img = process_image(fil)
        if img is not None:
            result = predict(img)
            confidence = round(result[0][0] * 100, 2)
            if result[0][0] > 0.5:
                prediction = f'**Positive**: Brain hemorrhage detected with {confidence}% confidence.'
                st.success(prediction)
            else:
                prediction = f'**Negative**: No brain hemorrhage detected with {100 - confidence}% confidence.'
                st.info(prediction)

            # Save prediction to history
            st.session_state["history"].append(prediction)

            # Display prediction history
            st.subheader("Prediction History")
            for idx, hist in enumerate(st.session_state["history"]):
                st.write(f"{idx + 1}. {hist}")

            # Plot confidence graph
            st.subheader("Confidence Levels")
            plot_confidence_graph(confidence)

            # Allow downloading results
            st.download_button(
                label="Download Results",
                data=save_results_to_file(st.session_state["history"]),
                file_name="prediction_results.txt",
                mime="text/plain"
            )
else:
    st.warning('Please upload an image to proceed.')