import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import tensorflow as tf

model = tf.keras.models.load_model('my_model.hdf5')

def import_and_predict(image_data, model):
    
        size = (75,75)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = image.convert('RGB')
        image = np.asarray(image)
        image = (image.astype(np.float32) / 255.0)

        img_reshape = image[np.newaxis,...]

        prediction = model.predict(img_reshape)
        
        return prediction

model = tf.keras.models.load_model('my_model.hdf5')

st.write("""
         # Cancer Prediction
         """
         )

st.write("This is a simple image classification web app to predict cancer")

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])
#
if file is None:
    st.text("You haven't uploaded an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) == 0:
        st.write("Not sick")
    elif np.argmax(prediction) == 1:
        st.write("Sick")
    else:
        st.write("")
    
    st.text("Probability (0: Not sick, 1: Sick)")
    st.write(prediction)