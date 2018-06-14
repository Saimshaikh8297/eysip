import cv2
import numpy as np
img = np.zeros((500,500,3), np.uint8)
coordinates=[[29.5, 250.0], [29.0, 250.0], [38.0, 385.0], [160.97999572753906, 417.5], [114.12354278564453, 338.9093322753906], [88.5, 259.0], [158.53448486328125, 202.6724090576172], [187.5, 38.5], [261.2481384277344, 121.8302230834961], [270.5, 243.0], [291.4565124511719, 422.2826232910156], [387.043701171875, 360.78155517578125], [343.0, 274.5], [362.0, 166.5]]

for i in coordinates:
    cv2.circle(img,(int(i[0]),int(i[1])),20,(255,255,255),thickness=-1)


#if c[0]>=coordinates[i][0]-10 and c[0]>=coordinates[i][0]+10 and c[1]>=coordinates[i][1] - 10
cap = cv2.VideoCapture("NEW VIDS & RESULTS/new_output11.mp4")
_,frame = cap.read()

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

lower_red = np.array([133, 107, 135])
upper_red = np.array([165, 255, 255])

mask = cv2.inRange(hsv, lower_red , upper_red)
res = cv2.bitwise_and(frame, frame , mask = mask)

gray = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
gray = cv2.cvtColor(gray,cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(gray, 30, 200)

(_,contours, _) = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
cnt = contours[0]
(x, y), radius = cv2.minEnclosingCircle(cnt)
center = (int(x), int(y))
#coordinates.append([x,y])
radius = int(radius)
cv2.circle(res, center, radius, (0, 255, 0), 2)
cv2.line(img, (int(x), int(y)), (int(x), int(y)), (255, 0, 0), 3)

while (1):
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([133, 107, 135])
    upper_red = np.array([165, 255, 255])

    mask = cv2.inRange(hsv, lower_red , upper_red)
    res = cv2.bitwise_and(frame, frame , mask = mask)

    gray = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(gray,cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)

    (_,contours, _) = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    cnt = contours[0]
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    coordinates.append([x,y])
    radius = int(radius)
    cv2.circle(res, center, radius, (0, 255, 0), 2)
    #print("area of circle = ")
    #print((3.14)*(radius*radius))

    if (3.14)*(radius*radius)<700:
        print("small circle")
        x = 0
        y = 0

    if int(x) != 0 and int(y) != 0 :
        cv2.line(img, (int(x), int(y)), (int(x), int(y)), (255, 0, 0), 3)
    elif int(x)==0 and int(y)==0:
        print("gap")
        #cv2.line(img, (int(old_x), int(old_y)), (int(old_x+1), int(old_y+1)), (255, 0, 0), 10)
    else:
        cv2.line(img, (int(x), int(y)), (int(x), int(y)), (255, 0, 0), 3)
    old_x = x
    old_y = y
    #print(old_x, old_y)




    #cv2.drawContours(res, [cnt], -1, (0, 255, 0), 3)

    #print(coordinates.__len__())
    cv2.imshow('frame', hsv)
    #cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    cv2.imshow("plot",img)


    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
#print(coordinates[::200])
cv2.imwrite("NEW VIDS & RESULTS/planterbot11_plot.png",img)
cv2.destroyAllWindows()
cap.release()
