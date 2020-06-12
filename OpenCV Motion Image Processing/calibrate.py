# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 12:28:52 2020

@author: tosri
"""
import numpy as np
import cv2
import time as t



calibrate=np.load('pencalibration.npy')
capture=cv2.VideoCapture(0)
pen_img=cv2.resize(cv2.imread('pen.jpg',1), (50,50)) #loading pen icon
eraser_img=cv2.resize(cv2.imread('eraser.jpg',1), (50,50)) #loading eraser icon
kernel=np.ones((5,5), np.uint8) #for .erode() and .dilate(), subject to change based on target object
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
canvas=None
background=cv2.createBackgroundSubtractorMOG2(detectShadows=False)
backthresh=600
switch='Pen'
last_switch=t.time()
x1,y1=0,0 #selecting the first points for contour, will later give them legitimate values
noisethresh=800 #setting a threshold for random noise such that the contour has to be larger than this to be considered as a contour.
wipethresh=40000 #target object contour area has to be higher than this constant in order to wipe the board.
clear=False #boolean variable that will change as we wipe the board, false means no wipe

while(1):
    _, frame=capture.read()
    frame=cv2.flip(frame, 1)
    if canvas is None:
        canvas=np.zeros_like(frame)
    topleft=frame[0:50, 0:50]
    figmask=background.apply(topleft)
    switchthresh=np.sum(figmask==255)
    #If we have a disruption greater than our backtrhesh and it's been one second since the last switch, switch the object from pen to eraser or vice versa
    if switchthresh>backthresh and (t.time()-last_switch)>1:
        last_switch=t.time()
        if switch=='Pen':
            switch='Eraser'
        else:
            switch='Pen'
    hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lowran=calibrate[0]
    highran=calibrate[1]
    binmask=cv2.inRange(frame, lowran, highran)
    binmask=cv2.erode(binmask, kernel, iterations=1)
    binmask=cv2.dilate(binmask, kernel, iterations=2)
    #creating and locating contours in the image
    contours, _ = cv2.findContours(binmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours and cv2.contourArea(max(contours, key=cv2.contourArea))>noisethresh:
        #running this code if and only if our contour area is bigger than our noise threshold
        c=max(contours, key=cv2.contourArea)
        x2,y2,w,h=cv2.boundingRect(c) #finding coordinates of bounding rectangle of contour
        # just an fyi, a contour is a 2D box with a certain amount of enclosed area
        area=cv2.contourArea(c) #getting area of our contour
        #If this is our first point, we save x2 and y2 as x1 and y1, respectively
        if x1==0 and y1==0:
            x1,y1=x2,y2
        else:
            if switch=='Pen':
                canvas=cv2.line(canvas, (x1,y1), (x2,y2), [255,0,0], 5)
            else:
                canvas=cv2.circle(canvas, (x2,y2),20,(0,0,0),-1)
        x1,y1=x2,y2 #we want our new starting point to build off of the old ending point
        if area>wipethresh: #we want to wipe the board if our contour area is bigger than our wiping threshold
            cv2.putText(canvas, 'Wiping the board! ^~^', (0,200), cv2.FONT_HERSHEY_SIMPLEX, (0,255,0), 4, cv2.LINE_AA)
            clear=True #only changes to false when we draw on the board
    else:
        x1,y1=0,0
        
    #### BETTER DRAWING ####
    _, binmask = cv2.threshold(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY), 20, 255, cv2.THRESH_BINARY)
    forward = cv2.bitwise_and(canvas, canvas, mask=binmask) #everything in the mask
    backwards=cv2.bitwise_and(frame, frame, cv2.bitwise_not(binmask)) #everything not in the mask
    frame = cv2.add(forward, backwards)
    
    
    if switch!='Pen':
        cv2.circle(frame, (x1,y1),20,(255,255,255), -1) #color in black so it looks like you erased the board
        frame[0:50, 0:50]=eraser_img
    else:
        frame[0:50,0:50]=pen_img
    cv2.imshow('image', frame)
    k=cv2.waitKey(5) & 0xFF
    if k==27:
        break
    if clear==True:
        t.sleep(1)
        canvas=None
        clear=False
    cv2.imshow('image', frame)
    k=cv2.waitKey(5) & 0xFF
    if k==27:
        break
cv2.destroyAllWindows()
capture.release()