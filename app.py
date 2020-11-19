# imports --------------
import streamlit as slit
from PIL import Image, ImageEnhance, ImageColor
import numpy as np
import cv2 
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import detect_mask_in_image
# end


# Serialized face detector
def face_mask_detection ():    
    slit.title("Face Mask Detection")
    user_choice = ['Image', 'Webcam']
    slit.set_option('deprecation.showfileUploaderEncoding', False)
    choice = slit.sidebar.selectbox('Mask Detection is on. ', user_choice)
    
    if choice =='Image':   
        slit.subheader('Detection on image')
        user_image_file = slit.file_uploader('Upload Image', type=['jpg','png']) # Image uploader 11/10/2020
        
        if user_image_file is not None:    
            user_image = Image.open(user_image_file)
            img = user_image.save('./images/out.jpg')
            saved_image_from_user = slit.image(user_image_file, caption = 'Your Image Uploaded Successfully', use_column_width = True)
            
            if slit.button('Process'):   
                slit.image(RGB_image , use_column_width=True)
                
    if choice == 'Webcam':    
        slit.subheader("Detection on user webcam")
        st.text("This feature will be available very soon Follow my git-hub @ CodeBalch25")
        
face_mask_detection()

# ---------------------------------------------------------------------------------------------


def face_mask_image():    
    global RGB_image
    
    print('[Info] Loading Face Detector Model...')
    proto_Path = os.path.sep.join(['face_detector', 'deploy.prototxt'])
    weights_Path = os.path.sep.join(['face_detection', 'res10_300x300_ssd_iter_140000.caffemodel'])
    
    net = cv2.dnn.readNet(proto_Path,weights_Path)
    
    
    print('[Info] loading face mask detector model........')
    # load the face mask detector model
    model = load_model('face_mask_detector.model')
    
    # load image input and grab the image spatial / dimensions
    
    image =  cv2.imread('./images/out.jpg')
    
    (h,w) = image.shape[:2]
    
    # construct a blob from the image 
    blob = cv2.dnn.blobFromImage(image, 1.0, (300,300), (104.0, 177.0, 123.0))
    
    # pass the blob through the network and get the face user face_detection
    
    print('[Info] computing Face Detection......please wait') 
    
    net.setInput(blob)
    
    user_detections = net.forward()
    
    # looping over user_detections 
    for i in range(0, user_detections.shape[2]):  
        # probability associated with user_detections
        
        confirm = user_detections[0,0,i,2]
        
        # filter weak user_detections by ensuring confirm is greater then min confirm
        
        if confirm > 0.5:   
            # compute the x, y coordinates
            
            coor_box = user_detections[0,0,i,3:7] * np.array([w,h,w,h])
            (startX, startY, endX, endY) = coor_box.astype('int')
            
            # ensure the coor_box are with in the dimentions of the frame. 
            (startX, startY) = (max(0,startX),max(0,startY))
            (endX , endY) = (min(w - 1, endX), min(h-1, endY))
            
            # extract the face ROI, convert , convert iy from BGR to Rgb channel
            #   and resize it to 224 x224, and preprocess it.   
            
            face_change = image[startY:endY, startX:endX]
            face_change = cv2.cvtColor(face_change, cv2.COLOR_BGB2RGB)            
            face_change = cv2.resize(face_change,(224,224))
            face_change = img_to_array(face_change)
            face_change = preprocess_input(face_change)
            face_change = np.expand_dims(face_change, axis=0)
            
            # pass the face_change through the model to determine if user has mask or not 
            
            (with_mask, with_out_mask) = model.predict(face_change)[0]
            
            #determin the class label and color used to draw coor_box retangle output frame
            
            label = "Mask" if with_mask > with_out_mask else "No Mask"
            
            color = (0,255,0) if label == "Mask" else (0,0,255)
            
            # Probability in label
            label = "{}: {:.2f}%".format(label, max(with_mask,with_out_mask) * 100)
            
            
            # label retangle output frame
            
            cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,color,2)
            
            cv2.retangle(image, (startX, startY), (endX, endY), color,2)
            
            RGB_image = cv2.cvrColor(image, cv2.COLOR_B2B2RGB)
            
face_mask_image()
            
            
            
            
            
        
          
        
    
    
    
    
   




