import numpy as np
import cv2
import time as t

def nothing(x):
    pass

capture=cv2.VideoCapture(0)
capture.set(3,1280)
capture.set(4,720)
cv2.namedWindow("Photo Capture")
cv2.createTrackbar("L-H", "Photo Capture", 0,179, nothing)
cv2.createTrackbar("L-S", "Photo Capture", 0,255, nothing)
cv2.createTrackbar("L-V", "Photo Capture", 0,255, nothing)
cv2.createTrackbar("U-H", "Photo Capture", 179,179, nothing)
cv2.createTrackbar("U-S", "Photo Capture", 255,255, nothing)
cv2.createTrackbar("U-V", "Photo Capture", 255,255, nothing)

while True:
    _, frame=capture.read()
    hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    l_h=cv2.getTrackbarPos("L-H", "Photo Capture")
    l_s=cv2.getTrackbarPos("L-S", "Photo Capture")
    l_v=cv2.getTrackbarPos("L-V", "Photo Capture")
    u_h=cv2.getTrackbarPos("U-H", "Photo Capture")
    u_s=cv2.getTrackbarPos("U-S", "Photo Capture")
    u_v=cv2.getTrackbarPos("U-V", "Photo Capture")
    lowrange=np.array([l_h,l_s,l_v])
    highrange=np.array([u_h, u_s, u_v])
    mask_b=cv2.inRange(hsv, lowrange, highrange)
    rgbmask=cv2.cvtColor(mask_b, cv2.COLOR_GRAY2BGR)
    res=cv2.bitwise_and(frame, frame, mask=mask_b)
    stackystack=np.hstack((rgbmask, frame))
    cv2.imshow('Photo Capture', cv2.resize(stackystack, None, fx=0.4, fy=0.4))
    key=cv2.waitKey(1)
    if key==27:
        break
    if key==ord('x'):
        array=[[l_h,l_s,l_v], [u_h, u_s, u_v]]
        print(array)
        np.save('pencalibration', array)
        break
capture.release()
cv2.destroyAllWindows()

