import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the pre-trained model
model = load_model('DenseNet_model.h5')

# Set image size for prediction
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Function to preprocess the image
def preprocess_image(img):
    img = img.resize((IMG_HEIGHT, IMG_WIDTH))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image
    return img_array

# Streamlit UI
st.title('Gastrointestinal Cancer Prediction with DenseNet121')

# Upload Image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open and display the image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    img_array = preprocess_image(img)

    # Make prediction
    prediction = model.predict(img_array)

    # Display prediction result
    if prediction[0] > 0.5:
        st.write("Prediction: **MSS (Microsatellite Stable)**")
    else:
        st.write("Prediction: **MSIMUT (Microsatellite Instable)**")
