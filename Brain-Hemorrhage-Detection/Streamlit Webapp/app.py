import tensorflow as tf
import streamlit as st
import numpy as np
import cv2
from PIL import Image

# st.set_option('deprecation.showfileUploaderEncoding', False)

# Load the model
try:
    model = tf.keras.models.load_model('final.h5')
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.title('Brain Hemorrhage Detection')

st.markdown("### Project Link: [http://localhost:8501](http://localhost:8501)")

fil = st.file_uploader('', type=['jpg', 'png', 'jpeg'])

def func():
    file_bytes = np.asarray(bytearray(fil.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    st.image(opencv_image, channels="RGB")

    # Convert image for TensorFlow processing
    fil.seek(0)  # Reset file pointer
    image = Image.open(fil)
    image = image.resize((130, 130))
    img = np.array(image)
    
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    ans = model.predict(img)
    return ans

if fil is not None:
    result = func()
    if result[0][0] > 0.5:
        st.write('**Positive : **', round(result[0][0], 2))
    else:
        st.write('Negative')
else:
    st.write('Please Upload an Image')
