import cv2
import numpy as np
import pickle




#import the xml file to detect the face
face_cascade=cv2.CascadeClassifier('/home/rubesh/opencv-3.4.0/data/haarcascades/haarcascade_frontalface_default.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")
labels={"person_name":1}
with open("labels.pickle",'rb') as f:
  og_labels=pickle.load(f)
  labels={v:k for k,v in og_labels.items()}
#capture the web video
cap=cv2.VideoCapture(0)
while(1):
  ret,frame=cap.read()
  gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
  faces= face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)#store the capture frame
  for(x,y,w,h) in faces:
    print(x,y,w,h)
    roi_gray = gray[y:y+h,x:x+w]
    roi_color = frame[y:y+h,x:x+w]

    id_,conf = recognizer.predict(roi_gray)
    if conf>=45 and conf <=85:
      print(id_)
      print(labels[id_])
      
    
    #img_item="sam.png"
    #cv2.imwrite(img_item,roi_color)

    color=(255,0,0)
    stroke =2
    end_cor_x = x+w
    end_cor_y = y+h
    cv2.rectangle(frame,(x,y),(end_cor_x,end_cor_y),color,stroke)
  cv2.imshow('frame',frame)
  x=cv2.waitKey(2)
 
  if  x == ord('x'):
    break
cap.release();
cv2.destroyAllWindows()
 
