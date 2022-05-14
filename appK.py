import cv2
from keras.models import load_model
from keras.preprocessing.image import load_img , img_to_array
import numpy as np
                                                                            #load model
model=load_model('face_mask_detection_model.h5')
# model accept below height and width of the image
img_width , img_height = 200 , 200
# Load the Cascade face Classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#Start web-cam
cap = cv2.VideoCapture(0)# for webcam
img_count_full = 0
#parameters for text
# font
font = cv2.FONT_HERSHEY_SIMPLEX
# org 
org = (1,1)
class_label = ' '
# fontScale 
fontScale = 1
# Blue color in BGR 
color = (255,0,0)
# Line thickness of 2 px 
thickness=2
#start reading images and prediction
while True:
    img_count_full+=1
    #read image from webcam
    res , clr_img = cap.read()
    #if response False the break the loop
    if res==False:
        break
# Convert to grayscale
    gray_img = cv2.cvtColor(clr_img , cv2.COLOR_BGR2GRAY)
# Detect the faces
    faces = face_cascade.detectMultiScale(gray_img , 1.1, 6)
#take face then predict class mask or not mask then draw rectangle and text then display image    
    img_cnt = 0
    for (x,y,w,h) in faces:
        org = (x-10, y-10)
        img_cnt +=1
        clr_face = clr_img[y:y+h , x:x+w]
        cv2.imwrite('input_faces/faces/%d%dface.jpg'%(img_count_full , img_cnt) , clr_face)
        img = load_img('input_faces/faces/%d%dface.jpg'%(img_count_full , img_cnt) , target_size = (img_width , img_height))
        
        img = img_to_array(img)
        img = np.expand_dims(img,axis=0)
        pred_p = model.predict(img)
        pred = np.argmax(pred_p)
        
        if pred == 0:
            print('User with mask = ',pred_p[0][0])
            class_label = 'Mask'
            color = (255,0,0)
            cv2.imwrite('input_faces/with_mask/%d%dface.jpg'%(img_count_full , img_cnt) , clr_face)
            
        else:
            print('User without mask = ',pred_p[0][1])
            class_label = 'No Mask'
            color = (0,255,0)
            cv2.imwrite('input_faces/without_mask/%d%dface.jpg'%(img_count_full , img_cnt) , clr_face)
            
        cv2.rectangle(clr_img , (x,y) , (x+w,y+h) , (0,0,255) , 3)
        cv2.putText(clr_img , class_label , org , font , fontScale , color , thickness , cv2.LINE_AA)
    cv2.imshow("LIVE FACE MASK DETECTION",clr_img)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
                                                                    # Release the VideoCapture object            
cap.release()
cv2.destroyAllWindows()