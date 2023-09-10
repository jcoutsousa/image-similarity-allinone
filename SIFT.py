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