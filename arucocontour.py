import imutils
import numpy as np
import cv2
import cv2.aruco as aruco
from warp_final import warpingfunc

flag_contour=0
cap = cv2.VideoCapture("new_video11.mov") # Capture video from camera

# Get the width and height of frame
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use the lower case
out = cv2.VideoWriter('new_output_aruco_contours_11.mp4', fourcc, 20.0, (width, height))
count = 0
flag=True
while(cap.isOpened()):
    print(" i am in first loop")
    ret, frame = cap.read()
    area=(frame.shape)
    frame_area =area[0]*area[1]
    print("Frame Area")
    print(frame_area)

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
    if flag_contour==0:
        print("i am in cont")
        #contour filtering part
        blurred = cv2.bilateralFilter(gray, 11, 17, 17)
        kernel = np.ones((5, 5), np.uint8)
        blurredopen = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
        blurredopen = cv2.morphologyEx(blurredopen, cv2.MORPH_OPEN, kernel)
        blurredclose = cv2.morphologyEx(blurredopen, cv2.MORPH_CLOSE, kernel)
        edged = cv2.Canny(blurredclose, 30, 200)
        cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        cntsSorted = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)[:1]
        for c in cntsSorted:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.01 * peri, True)
            # if our approximated contour has four points, then
            # we can assume that we have found our screen
            if len(approx) == 4:
                contour_area = (cv2.contourArea(c))
                print("Area Percent")
                # print(cv2.contourArea(c))
                areapercent = (contour_area/frame_area)*100
                print(areapercent)
                if areapercent>25 :
                    print("Arena Found")
                    screenCnt = approx
                    contours = screenCnt
                    x1 = contours[0][0][0]
                    y1 = contours[0][0][1]
                    x2 = contours[1][0][0]
                    y2 = contours[1][0][1]
                    x3 = contours[2][0][0]
                    y3 = contours[2][0][1]
                    x4 = contours[3][0][0]
                    y4 = contours[3][0][1]
                    # print("HII")
                    print((x1, y1))
                    print((x2, y2))
                    print((x3, y3))
                    print((x4, y4))
                    flag_contour=1
                break

    #aruco part
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    gray = aruco.drawDetectedMarkers(gray, corners, ids)
    # cv2.imshow('frame', gray)
    for a in corners:
        tlx = a[0][0][0]
        tly = a[0][0][1]
        trix = a[0][1][0]
        triy = a[0][1][1]
        blx = a[0][3][0]
        bly = a[0][3][1]
        brx = a[0][2][0]
        bry = a[0][2][1]

    if ret == True and (tlx, tly, trix, triy, blx,bly,brx,bry)!=(0,0,0,0,0,0,0,0) :
        out.write(frame)

        cv2.imshow('frame',frame)
        # print((tlx, tly, trix, triy, blx,bly,brx,bry))
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
print("i am out")
warpingfunc(x1, y1, x2, y2, x3, y3, x4, y4)
out.release()
cap.release()
cv2.destroyAllWindows()


# print(count)