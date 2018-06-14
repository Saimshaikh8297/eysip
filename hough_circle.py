import cv2
import numpy as np


#img = cv2.imread('planterbot111_plot.png')
img = np.zeros((500,500,3), np.uint8)



coordinates=[[29.5, 250.0],[29.0, 250.0], [38.0, 385.0], [160.97999572753906, 417.5], [114.12354278564453, 338.9093322753906], [88.5, 259.0], [158.53448486328125, 202.6724090576172], [187.5, 38.5], [261.2481384277344, 121.8302230834961], [270.5, 243.0], [291.4565124511719, 422.2826232910156], [387.043701171875, 360.78155517578125], [343.0, 274.5], [362.0, 166.5]]
#image_array = np.zeros((500,500), np.uint8)
for i in coordinates:
    cv2.circle(img,(int(i[0]),int(i[1])),20,(255,255,255),thickness=-1)
circle_radius = 8
d=0
cv2.imshow("plot",img)
for i in coordinates:
    i[0] = int(i[0])
    i[1] = int(i[1])

    roi = img[i[1] - (3* circle_radius): i[1]+ (3*circle_radius), i[0]- (3*circle_radius): i[0]+ (3*circle_radius)]
    roi = roi.reshape(int(roi.size/3),3)
    #print(roi)
    cv2.imshow("circle", roi)

    if [255,0,0] in roi.tolist():
        print("yes")

    else:
        print("no")

    # mask = np.zeros(image_array.shape, dtype=np.uint8)
    # mask = cv2.circle(img, (i[0],i[1]), circle_radius, (0, 255, 255), -1, 8, 0)

    # Apply mask (using bitwise & operator)
    # result_array = image_array & mask

    # Crop/center result (assuming max_loc is of the form (x, y))
    # result_array = result_array[i[1] - circle_radius:i[1] + circle_radius,i[0] - circle_radius:i[0] + circle_radius, :]

    # d=d+1

# cv2.imshow('detected circles',cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()