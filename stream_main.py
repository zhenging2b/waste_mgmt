import streamlit as st
from PIL import Image
from clf_test import predict


st.title("Zheng Ing's Image Classification  for Waste Management App")
st.write("")


file_up = st.file_uploader("Upload an image", type="jpg")
if file_up is not None:
    image = file_up.read()
    st.image(image, caption='Uploaded Image.', use_container_width=True)
    st.write("")
    st.write("Just a second...")
    st.write("Self defined CNN")
    st.write(predict(image))
    # st.write("VGG")
    # st.write(predict_vgg(image))