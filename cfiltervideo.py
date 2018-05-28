import cv2
import imutils

# image = cv2.imread("test.jpg")
cap = cv2.VideoCapture("vid3.mp4")
while(1):
    ret, image = cap.read()
    resized = imutils.resize(image, width=600)
    ratio = image.shape[0] / float(resized.shape[0])
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 11, 17, 17)
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # ret,thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_OTSU)
    edged = cv2.Canny(blurred, 30, 200)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cntsSorted = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)
    for c in cntsSorted:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if our approximated contour has four points, then
        # we can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break
    print(screenCnt)
    cv2.drawContours(resized, [screenCnt], -1, (0, 255, 0), 3)
    cv2.imshow('frame', resized)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
