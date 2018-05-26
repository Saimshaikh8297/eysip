import numpy as np
import cv2
print(cv2.__version__)
cap = cv2.VideoCapture("vid1.mp4")

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # print(frame.shape) #480x640
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # gray1 = clahe.apply(gray)
    # gray = clahe.apply(gray)
    # gray1 = cv2.fastNlMeansDenoising(gray1, None,5, 21, 7)
    # cv2.imshow('frameclahe',gray1)
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break