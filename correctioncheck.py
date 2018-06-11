import numpy as np
import cv2
import cv2.aruco as aruco

cap = cv2.VideoCapture("test6.mp4") # Capture video from camera

# Get the width and height of frame
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use the lower case
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width, height))
count = 0
flag=True
while(cap.isOpened()):
    ret, frame = cap.read()

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

    if ret == True and (tlx, tly, trix, triy, blx,bly,brx,bry)==(0,0,0,0,0,0,0,0) :
        out.write(frame)

        cv2.imshow('frame',frame)
        flag=True

        if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
            break
    else:
        if (flag == True):
            count+=1
            flag=False

    if (cv2.waitKey(1) & 0xFF) == ord('q'):  # Hit `q` to exit
        break

# Release everything if job is finished
out.release()
cap.release()
cv2.destroyAllWindows()
print(count)