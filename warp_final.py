import imutils
import numpy as np
import cv2

opframe=(500,500)
cap = cv2.VideoCapture("new_output_aruco_contours_11.mp4")
width = 500
height = 500
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use the lower case
out = cv2.VideoWriter('new_output11.mp4', fourcc, 20.0, (width, height))

def warpingfunc(x1,y1,x2,y2,x3,y3,x4,y4):
    while(1):
        ret, image = cap.read()
        mask = np.zeros(image.shape, dtype=np.uint8)
        roi_corners = np.array([[(x1, y1), (x2, y2), (x3, y3), (x4,y4)]], dtype=np.int32)
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
        cv2.fillPoly(mask, roi_corners, ignore_mask_color)
        masked_image = cv2.bitwise_and(image, mask)

        pts1 = np.float32([(x3, y3), (x2, y2), (x4, y4), (x1,y1)])
        pts2 = np.float32([[0, 0], [500, 0], [0, 500], [500, 500]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(masked_image, M, (500, 500))
        if ret == True:
            print("hi")
            out.write(dst)

            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        else:
            break

    out.release()
    cap.release()
    cv2.destroyAllWindows()
