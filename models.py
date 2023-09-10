from PIL import Image
import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from matplotlib import pyplot as plt

#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg

from keras.applications.vgg16 import VGG16
from sklearn.metrics.pairwise import cosine_similarity

# Configuring the VGG16 Model for Image Embedding Extraction
from keras.applications import VGG19

# weights='imagenet' parameter specifies that the model should be initialized with pre-trained weights from the ImageNet dataset
# include_top=False parameter indicates that the top dense layers of the model, which are responsible for classification, should not be included.
# parameter pooling='max' operation is useful for reducing the size of the feature maps while retaining the most important information

VGG19 = VGG19(weights='imagenet', include_top=False,
              pooling='max', input_shape=(224, 224, 3))

# print the summary of the model's architecture.
VGG19.summary()

# Freezing the VGG16 Model Layers for Transfer Learning
# For each layer in the model, we need to specify that we don’t need additional training. We will instead use the pre-set parameters of the VGG16 model, which was trained by default with the ImageNet dataset.

for model_layer in VGG19.layers:
    model_layer.trainable = False


def clip_similarity(image_original, image_received):
    import torch
    import clip

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

class vgg():

    # Defining the functions for Preprocessing the Image Data for Model Input
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

    def get_image_embeddings(object_image: image):
        """
          -----------------------------------------------------
          convert image into 3d array and add additional dimension for model input
          -----------------------------------------------------
          return embeddings of the given image
        """

        image_array = np.expand_dims(image.img_to_array(object_image), axis=0)
        image_embedding = VGG19.predict(image_array)

        return image_embedding

    def get_similarity_score(first_image: str, second_image: str):
        """
            -----------------------------------------------------
            Takes image array and computes its embedding using VGG16 model.
            -----------------------------------------------------
            return embedding of the image

        """

        first_image = vgg.load_image(first_image)
        second_image = vgg.load_image(second_image)

        first_image_vector = vgg.get_image_embeddings(first_image)
        second_image_vector = vgg.get_image_embeddings(second_image)

        similarity_score = cosine_similarity(first_image_vector, second_image_vector).reshape(1, )

        return similarity_score

    def vgg_similarity(uploaded_file_original, uploaded_file):
        if (uploaded_file and uploaded_file_original) is not None:
            st.title("Outcome")
            similarity_score = vgg.get_similarity_score(uploaded_file_original, uploaded_file)
            if similarity_score >= 0.75:
                #print("It is similar\n", similarity_score)
                st.write("It is similar\n", str(similarity_score))
            else:
                #print("Not the same image", similarity_score)
                st.write("Not the same image\n", str(similarity_score))
        else:
            print('No images to process')
            st.title("Outcome")
            st.write('No images to process')

def sift(image_original, image_received):
    import cv2

    if image_original and image_received is not None:
        st.title("Outcome")
        "for both of these images, we are going to generate the SIFT features. " \
        "First, we have to construct a SIFT object. " \
        "We first create a SIFT object using sift_create and then use the function detectAndCompute to get the keypoints. " \
        "It will return two values – the keypoints and the sift descriptors."

        # keypoint descriptor
        # reading image / # Convert the file to an opencv image.
        img1_bytes = np.asarray(bytearray(image_original.read()), dtype=np.uint8)
        img1_original = cv2.imdecode(img1_bytes, 1)
        img2_bytes = np.asarray(bytearray(image_received.read()), dtype=np.uint8)
        img2_received = cv2.imdecode(img2_bytes, 1)

        # Convert the training image to RGB
        training_image = cv2.cvtColor(img1_original, cv2.COLOR_BGR2RGB)

        # Convert the training image to gray scale
        img1 = cv2.cvtColor(training_image, cv2.COLOR_RGB2GRAY)

        # Convert the tested image to RGB
        tested_image = cv2.cvtColor(img2_received, cv2.COLOR_BGR2RGB)

        # Convert the training image to gray scale
        img2 = cv2.cvtColor(tested_image, cv2.COLOR_RGB2GRAY)

        #img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        #img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # keypoints - sift
        sift = cv2.SIFT_create()

        keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
        keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

        print(len(keypoints_1), len(keypoints_2))
        print(len(descriptors_1), len(descriptors_2))

        "Let’s try and match the features from image 1 with features from image 2. " \
        "We will be using the function match() from the BFmatcher (brute force match) module. " \
        "Also, we will draw lines between the features that match both images. " \
        "This can be done using the drawMatches function in OpenCV python."

        # feature matching
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches_flann = flann.knnMatch(descriptors_1, descriptors_2, k=2)


        # Apply ratio test based on Low test
        bestMatches = []
        for match1, match2 in matches_flann:
            if match1.distance < 0.80 * match2.distance:
                bestMatches.append(match1)

        print(len(bestMatches))
        MIN_MATCH_COUNT = 0.05*len(matches_flann)

        if len(bestMatches) > MIN_MATCH_COUNT:
            src_pts = np.float32([keypoints_1[m.queryIdx].pt for m in bestMatches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints_2[m.trainIdx].pt for m in bestMatches]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
            h, w = img1.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
            print("Enough matches are found: {}/{}".format(len(bestMatches), MIN_MATCH_COUNT))
            st.write("Enough matches are found: {}/{}".format(len(bestMatches), MIN_MATCH_COUNT))
        else:
            print("Not enough matches are found: {}/{}".format(len(bestMatches), MIN_MATCH_COUNT))
            st.write("Not enough matches are found: {}/{}".format(len(bestMatches), MIN_MATCH_COUNT))
            matchesMask = None

        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matchesMask,  # draw only inliers
                           flags=2)
        img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, bestMatches, None, **draw_params)
        st.image(img3, caption='Comparison', width=560)
