import torch
import clip
from PIL import Image
import streamlit as st
import numpy as np

if __name__ == '__main__':

    st.header('ShipSnap CLIP Model')

    image_original = st.file_uploader('Upload Original Image', type = ['jpg','png','jpeg'])

    image_received = st.file_uploader('Upload Image', type=['jpg', 'png', 'jpeg'])


    if image_original and image_received is not None:
        st.image(image_original, caption='Original Image')
        st.image(image_received, caption='Image')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)

        image1 = image_original
        image2 = image_received


        cos = torch.nn.CosineSimilarity(dim=0)

        image1_preprocess = preprocess(Image.open(image1)).unsqueeze(0).to(device)
        image1_features = model.encode_image(image1_preprocess)

        image2_preprocess = preprocess(Image.open(image2)).unsqueeze(0).to(device)
        image2_features = model.encode_image(image2_preprocess)

        similarity = cos(image1_features[0], image2_features[0]).item()
        similarity = (similarity + 1) / 2
        print("Image similarity", similarity)

        st.title("Outcome")
        st.write("Image similarity", similarity)

    else:
        st.write("No images to compare")
