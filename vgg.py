import streamlit as st

import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from keras.applications.vgg16 import VGG16
from sklearn.metrics.pairwise import cosine_similarity

#Configuring the VGG16 Model for Image Embedding Extraction
from keras.applications import VGG19
#weights='imagenet' parameter specifies that the model should be initialized with pre-trained weights from the ImageNet dataset
#include_top=False parameter indicates that the top dense layers of the model, which are responsible for classification, should not be included.
#parameter pooling='max' operation is useful for reducing the size of the feature maps while retaining the most important information

VGG19 = VGG19(weights='imagenet', include_top=False,
              pooling='max', input_shape=(224, 224, 3))

# print the summary of the model's architecture.
VGG19.summary()

#Freezing the VGG16 Model Layers for Transfer Learning
#For each layer in the model, we need to specify that we don’t need additional training. We will instead use the pre-set parameters of the VGG16 model, which was trained by default with the ImageNet dataset.

for model_layer in VGG19.layers:
  model_layer.trainable = False

#Defining the functions for Preprocessing the Image Data for Model Input
def load_image(image_path):
    """
        -----------------------------------------------------
        Process the image provided.
        - Resize the image
        -----------------------------------------------------
        return resized image
    """

    input_image = Image.open(image_path)
    resized_image = input_image.resize((224, 224))

    return resized_image

def get_image_embeddings(object_image : image):

    """
      -----------------------------------------------------
      convert image into 3d array and add additional dimension for model input
      -----------------------------------------------------
      return embeddings of the given image
    """

    image_array = np.expand_dims(image.img_to_array(object_image), axis = 0)
    image_embedding = VGG19.predict(image_array)

    return image_embedding

def get_similarity_score(first_image : str, second_image : str):
    """
        -----------------------------------------------------
        Takes image array and computes its embedding using VGG16 model.
        -----------------------------------------------------
        return embedding of the image

    """

    first_image = load_image(first_image)
    second_image = load_image(second_image)

    first_image_vector = get_image_embeddings(first_image)
    second_image_vector = get_image_embeddings(second_image)

    similarity_score = cosine_similarity(first_image_vector, second_image_vector).reshape(1,)

    return similarity_score

def show_image(image_path):
  image = mpimg.imread(image_path)
  imgplot = plt.imshow(image)
  plt.show()

if __name__ == '__main__':
    st.header('ShipSnap: Streamlined delivery verification with visual matching')

    uploaded_file_original = st.file_uploader('Upload original file')


    uploaded_file = st.file_uploader('Upload the file')


    if (uploaded_file and uploaded_file_original) is not None:
        st.title("Outcome")
        similarity_score = get_similarity_score(uploaded_file_original, uploaded_file)
        if similarity_score >= 0.75:
            print("It is similar\n", similarity_score)
            st.write("It is similar\n", str(similarity_score))
            image_original = st.image(uploaded_file_original, caption='Original Image')
            image = st.image(uploaded_file, caption='Image')
        else:
            print("Not the same image", similarity_score)
            st.write("Not the same image\n", str(similarity_score))
            image_original = st.image(uploaded_file_original, caption='Original Image')
            image = st.image(uploaded_file, caption='Image')
    else:
        print('No images to process')
        st.title("Outcome")
        st.write('No images to process')