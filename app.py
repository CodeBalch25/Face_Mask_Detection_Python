# imports --------------
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import detect_mask_in_image
# end


# Serialized face detector
def face_mask_detection():
    st.title("Face Mask Detection")
    user_choice = ['Image', 'Webcam']
    st.set_option('deprecation.showfileUploaderEncoding', False)
    choice = st.sidebar.selectbox('Mask Detection is on. ', user_choice)

    if choice == 'Image':
        st.subheader('Detection on image')
        user_image_file = st.file_uploader('Upload Image', type=['jpg', 'png']) # Image uploader

        if user_image_file is not None:
            user_image = Image.open(user_image_file)
            img = user_image.save('./images/out.jpg')
            saved_image_from_user = st.image(user_image_file, caption='Your Image Uploaded Successfully', use_column_width=True)

            if st.button('Process'):
                st.image(RGB_image, use_column_width=True)

    if choice == 'Webcam':
        st.subheader("Detection on user webcam")
        st.text("This feature will be available very soon! Follow my GitHub @ CodeBalch25")

face_mask_detection()

# ---------------------------------------------------------------------------------------------


def face_mask_image():
    global RGB_image

    print('[Info] Loading Face Detector Model...')
    proto_Path = os.path.sep.join(['face_detector', 'deploy.prototxt'])
    weights_Path = os.path.sep.join(['face_detector', 'res10_300x300_ssd_iter_140000.caffemodel'])

    net = cv2.dnn.readNet(proto_Path, weights_Path)


    print('[Info] Loading face mask detector model...')
    # load the face mask detector model
    model = load_model('face_mask_detector.model')

    # load image input and grab the image spatial dimensions

    image = cv2.imread('./images/out.jpg')

    (h, w) = image.shape[:2]

    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and get the face user face_detection

    print('[Info] Computing Face Detection...please wait')

    net.setInput(blob)

    user_detections = net.forward()

    # looping over user_detections
    for i in range(0, user_detections.shape[2]):
        # probability associated with user_detections

        confidence = user_detections[0, 0, i, 2]

        # filter weak user_detections by ensuring confidence is greater than minimum confidence

        if confidence > 0.5:
            # compute the x, y coordinates

            coor_box = user_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = coor_box.astype('int')

            # ensure the coor_box are within the dimensions of the frame.
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert from BGR to RGB channel
            # and resize it to 224x224, and preprocess it.

            face_change = image[startY:endX, startX:endX]
            face_change = cv2.cvtColor(face_change, cv2.COLOR_BGR2RGB)
            face_change = cv2.resize(face_change, (224, 224))
            face_change = img_to_array(face_change)
            face_change = preprocess_input(face_change)
            face_change = np.expand_dims(face_change, axis=0)

            # pass the face_change through the model to determine if user has mask or not

            (with_mask, without_mask) = model.predict(face_change)[0]

            # determine the class label and color used to draw bounding box rectangle on output frame

            label = "Mask" if with_mask > without_mask else "No Mask"

            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # Probability in label
            label = "{}: {:.2f}%".format(label, max(with_mask, without_mask) * 100)


            # draw label and rectangle on output frame

            cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

            RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

face_mask_image()
