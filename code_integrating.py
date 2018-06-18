import imutils
import numpy as np
import cv2
import cv2.aruco as aruco

img = np.zeros((500,500,3), np.uint8)


def aruco_coordinates(file_name):

    ids = None
    flag_contour=0
    cap = cv2.VideoCapture(file_name) # Capture video from camera

    while(cap.isOpened() and (flag_contour==0 or ids== None)):
        ret, frame = cap.read()

        area=(frame.shape)
        frame_area =area[0]*area[1]
        print("Frame Area")
        print(frame_area)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if (ids== None):

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            gray_aruco = cv2.GaussianBlur(gray, (5, 5), 0)
            aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
            parameters = aruco.DetectorParameters_create()
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_aruco, aruco_dict, parameters=parameters)
            print("id of aruco = ")
            print(ids.flatten())


        if flag_contour==0:
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
                        flag_contour=1
                        if flag_contour==1 and ids != None:
                            break
    #print("i am out")
    cap.release()
    return ((ids.flatten()),contours)

def filter_top_of_robot(frame):
    #print("I am in filter_top_of_robot")


    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([124, 112, 171])
    upper_red = np.array([148, 193, 255])

    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    gray = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)

    (_, contours, _) = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    if contours.__len__()!=0:
        cnt = contours[0]
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        # coordinates.append([x,y])
        radius = int(radius)
        cv2.circle(res, center, radius, (0, 255, 0), 2)
        # print("area of circle = ")
        # print((3.14)*(radius*radius))

        if (3.14) * (radius * radius) < 700:
            #print("small circle")
            x = 0
            y = 0

        if int(x) != 0 and int(y) != 0:
            cv2.line(img, (int(x), int(y)), (int(x), int(y)), (255, 255, 255), 3)
        elif int(x) == 0 and int(y) == 0:
            print("gap")
            # cv2.line(img, (int(old_x), int(old_y)), (int(old_x+1), int(old_y+1)), (255, 0, 0), 10)
        else:
            cv2.line(img, (int(x), int(y)), (int(x), int(y)), (255, 255, 255), 3)
    else:
        print("no contour found bro")

    #cv2.imshow('res', res)
    cv2.imshow("Plot", img)
    cv2.waitKey(1)


def warping(image,contours):
    #print("I am in warping")


    x1 = contours[0][0][0]
    y1 = contours[0][0][1]
    x2 = contours[1][0][0]
    y2 = contours[1][0][1]
    x3 = contours[2][0][0]
    y3 = contours[2][0][1]
    x4 = contours[3][0][0]
    y4 = contours[3][0][1]
    # print("HII")
    #print((x1, y1))
    #print((x2, y2))
    #print((x3, y3))
    #print((x4, y4))

    mask = np.zeros(image.shape, dtype=np.uint8)
    roi_corners = np.array([[(x1, y1), (x2, y2), (x3, y3), (x4, y4)]], dtype=np.int32)
    channel_count = image.shape[2]
    ignore_mask_color = (255,) * channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)

    pts1 = np.float32([(x3, y3), (x2, y2), (x4, y4), (x1, y1)])
    pts2 = np.float32([[0, 0], [500, 0], [0, 500], [500, 500]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(masked_image, M, (500, 500))
    cv2.imshow("Warped", dst)
    cv2.waitKey(1)

    return dst


def deleteframes(team_id,file_name,contours,flag = True, count = 0):

    name = "team_id_" + str(team_id[0]) + ".png"
    print(name)
    cap = cv2.VideoCapture(file_name)

    while(cap.isOpened()):
        #print("I am in deleteframes")
        ret, image = cap.read()
        if ret == False:
            break

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
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
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

        if ret == True and (tlx, tly, trix, triy, blx,bly,brx,bry)!=(0,0,0,0,0,0,0,0):
            flag = True
            #print("printing frame" + str(count))
            #count += 1

            warped_frame = warping(image,contours)

            filter_top_of_robot(warped_frame)

        elif (tlx, tly, trix, triy, blx,bly,brx,bry)==(0,0,0,0,0,0,0,0):
            if flag==True:
                count += 1
                flag = False

        cv2.imshow("Original", image)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()
    #print("i am out of delete_frames")
    cv2.imwrite(name,img)
    return count





##############################################################################################################################################################
##############################################################################################################################################################


file_name = "new_video_5_1.mov"
(team_id, coordinates) = aruco_coordinates(file_name)
# x1 = coordinates[0][0][0]
# y1 = coordinates[0][0][1]
# x2 = coordinates[1][0][0]
# y2 = coordinates[1][0][1]
# x3 = coordinates[2][0][0]
# y3 = coordinates[2][0][1]
# x4 = coordinates[3][0][0]
# y4 = coordinates[3][0][1]
# print("Team_id = " + str(team_id))
# print((x1, y1))
# print((x2, y2))
# print((x3, y3))
# print((x4, y4))
count =deleteframes(team_id,file_name,coordinates)

print("Count = " + str(count))


