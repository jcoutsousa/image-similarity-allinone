import streamlit as st
import models


if __name__ == '__main__':

    # Setting page layout
    st.set_page_config(
        page_title="ShipSnap - Model Evaluation",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.header('ShipSnap Models')

    # Sidebar
    st.sidebar.header("ML Model Config")

    # Model Options
    model_type = st.sidebar.radio(
        "Select Task", ['SIFT', 'VGG19', 'CLIP'])

    #image_original = str(settings.DEFAULT_IMAGE)
    #image_received = str(settings.DEFAULT_DETECT_IMAGE)
    col1, col2 = st.columns(2)

    with col1:
        image_original = st.file_uploader('Upload Original Image', type=['jpg', 'png', 'jpeg'])
        if image_original is not None:
            st.image(image_original, caption='Original Image', width=240)

    with col2:
        image_received = st.file_uploader('Upload Image', type=['jpg', 'png', 'jpeg'])
        if image_received is not None:
            st.image(image_received, caption='Image', width=240)

    if model_type == 'CLIP':
        models.clip_similarity(image_original, image_received)
    elif model_type == 'VGG19':
        models.vgg.vgg_similarity(image_original, image_received)
    elif model_type == 'SIFT':
        models.sift(image_original, image_received)
    else:
        st.write("Not yet implemented")





