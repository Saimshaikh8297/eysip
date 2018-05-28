import cv2
import imutils
import numpy as np

# image = cv2.imread("test.jpg")
firstframe=False
cap = cv2.VideoCapture("vid5.mp4")
while(1):
    ret, image = cap.read()
    resized = imutils.resize(image, width=600)
    if(firstframe == False):

        ratio = image.shape[0] / float(resized.shape[0])
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.bilateralFilter(gray, 11, 17, 17)
        kernel = np.ones((5, 5), np.uint8)

        ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_OTSU)
        blurredopen = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        blurredclose = cv2.morphologyEx(blurredopen, cv2.MORPH_CLOSE, kernel)

        # blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        edged = cv2.Canny(blurredclose, 30, 200)
        # blurredopen = cv2.morphologyEx(edged, cv2.MORPH_OPEN, kernel)
        # blurredclose = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

        cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        cntsSorted = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)
        for c in cntsSorted:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.01 * peri, True)

            # if our approximated contour has four points, then
            # we can assume that we have found our screen
            if len(approx) == 4:
                screenCnt = approx
                break
        # print(screenCnt)
        contours=screenCnt
        firstframe=True




    # print(contours)
    # cropped = resized[155:582, 26:139]
    # cv2.imshow('framecrop', cropped)
    cv2.drawContours(resized, [contours], -1, (0, 255, 0), 3)
    cv2.imshow('frame', resized)
    x1 = contours[0][0][0]
    y1 = contours[0][0][1]
    print(x1)
    x2 = contours[1][0][0]
    y2 = contours[1][0][1]
    x3 = contours[2][0][0]
    y3 = contours[2][0][1]
    x4 = contours[3][0][0]
    y4 = contours[3][0][1]
    mask = np.zeros(resized.shape, dtype=np.uint8)
    roi_corners = np.array([[(x1, y1), (x2, y2), (x3, y3), (x4,y4)]], dtype=np.int32)
    channel_count = resized.shape[2]
    ignore_mask_color = (255,) * channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    masked_image = cv2.bitwise_and(resized, mask)
    cv2.imshow("crop",masked_image)
    cv2.imshow('frame1', blurred)
    cv2.imshow('frame2', edged)
    cv2.imshow('frame3', thresh)
    # cv2.imshow('frame4', blurredopen)
    cv2.imshow('frame5', blurredclose)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
