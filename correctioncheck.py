import numpy as np
import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt

opframe=(1280,720)
cap = cv2.VideoCapture("samplevideo.mp4")
fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # 'x264' doesn't work
out = cv2.VideoWriter('tstoutput.mp4',fourcc, 29.0, opframe, True)
count = 0
while (True):
    ret, frame = cap.read()
    # opframe = frame.shape
    # opframe = opframe[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tl = 0
    tr = 0
    bl = 0
    br = 0
    tlx = 0
    tly = 0
    blx = 0
    bly = 0
    trix = 0
    triy = 0
    brx = 0
    bry = 0
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    gray = aruco.drawDetectedMarkers(gray, corners, ids)
    cv2.imshow('frame', gray)
    for a in corners:
        tlx = a[0][0][0]
        tly = a[0][0][1]
        trix = a[0][1][0]
        triy = a[0][1][1]
        blx = a[0][3][0]
        bly = a[0][3][1]
        brx = a[0][2][0]
        bry = a[0][2][1]


    if ret == True:
        # writeframe = cv2.flip(dst, 0)
        out.write(gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
out.release()
print(count)
