# -*- coding: utf-8 -*-
"""
Created on Sat May 23 20:25:01 2020

@author: tosri
"""

import cv2
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade=cv2.CascadeClassifier('haarcascade_smile.xml')


def detect_smiles(gray, frame):
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h), (255,0,0), 2)
        roi_gray=gray[y:y+h, x:x+w]
        roi_color=frame[y:y+h, x:x+h]
        eyes=eye_cascade.detectMultiScale(roi_gray, 1.1,3)
        for(eye_x, eye_y, eye_w, eye_h) in eyes:
            cv2.rectangle(roi_color, (eye_x,eye_y), (eye_x+eye_w,eye_y+eye_h), (0,255,0), 2)
        smiles=smile_cascade.detectMultiScale(roi_gray,3.0,6)
        for (sx,sy,sw,sh) in smiles:
            cv2.rectangle(roi_color, (sx,sy), (sx+sw,sy+sh), (0,0,255), 2)
    return frame

capture=cv2.VideoCapture(0)
while True:
    _, frame = capture.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas=detect_smiles(gray, frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1)&0xFF == ord('f'):
        break
capture.release()
cv2.destroyAllWindows()