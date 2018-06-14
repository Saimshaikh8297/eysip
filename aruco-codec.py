import numpy as np
import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt


img = np.zeros((1080,1920,3), np.uint8)
#cap = cv2.VideoCapture(1)
cxold=0
cyold=0

while (True):
    # Capture frame-by-frame
    #ret, frame = cap.read()
    # print(frame.shape) #480x640
    # Our operations on the frame come here
    frame = cv2.imread("singlemarkersoriginal.png")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    # gray = cv2.fastNlMeansDenoising(gray, None, 5, 21, 7)
    gray = cv2.GaussianBlur(gray,(5,5),0)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()

    # print(parameters)

    '''    detectMarkers(...)
        detectMarkers(image, dictionary[, corners[, ids[, parameters[, rejectedI
        mgPoints]]]]) -> corners, ids, rejectedImgPoints
        '''
    # lists of ids and the corners beloning to each id
    tl=0
    tr=0
    bl=0
    br=0
    tlx=0
    tly=0
    blx=0
    bly=0
    trix=0
    triy=0
    brx=0
    bry=0
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    for a in corners:
        tlx=a[0][0][0]
        tly=a[0][0][1]
        trix=a[0][1][0]
        triy = a[0][1][1]
        blx=a[0][3][0]
        bly = a[0][3][1]
        brx=a[0][2][0]
        bry = a[0][2][1]

    tl=(tlx,tly)
    br=(brx,bry)

    cv2.rectangle(gray,tl,br,(255,0,0),5)
    cx=(tlx+brx)/2
    cy=(tly+bry)/2
    print((cxold,cyold))

    if int(cx) != 0 and int(cy) != 0 and int(cxold==0) and int(cyold==0):
        cv2.line(img, (int(cx), int(cy)), (int(cx), int(cy)), (255, 0, 0), 10)
    elif int(cx)==0 and int(cy)==0:
        cv2.line(img, (int(cxold), int(cyold)), (int(cxold+1), int(cyold+1)), (255, 0, 0), 10)
    else:
        cv2.line(img, (int(cxold), int(cyold)), (int(cx), int(cy)), (255, 0, 0), 10)

    cv2.line(gray,(int(cx),int(cy)),(int(cx),int(cy)),(255,0,0),10)
    if int(cx)!=0 and int(cy)!=0:
        cxold=cx
        cyold=cy
    # cv2.circle(gray,(int(cx),int(cy)) , 3, (255, 0, 0), -1)
    gray = aruco.drawDetectedMarkers(gray, corners,ids)

    # cv2.rectangle(gray,)

    # print(rejectedImgPoints)
    # Display the resulting frame
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
#cap.release()
#cv2.imwrite("plot.jpg",img)
cv2.destroyAllWindows()