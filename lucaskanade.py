import numpy as np
import cv2
cap = cv2.VideoCapture("001_output.mp4")
img1 = np.zeros((1280,720,3), np.uint8)
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0,255,(100,3))
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
old_gray = cv2.GaussianBlur(old_gray, (5, 5), 0)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
l=[]
l1=[]
while(1):
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray=cv2.GaussianBlur(frame_gray,(5,5),0)
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        # if(i==0):
        #     a, b = new.ravel()
        #     print("A,B"+ str(i))
        #     print((a, b))
        #     c, d = old.ravel()
        #     print("C,D")
        #     print((c, d))
        #     mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        #     frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
        #     img1 = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        #     frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
        a,b = new.ravel()
        c,d = old.ravel()
        if(abs(a-c)>0.05 and abs(b-d)>0.05):
            l.append((a,b))
            l1.append((c,d))
    if  l:
        (a, b) = l[0]
        print((a, b))
        (c, d) = l1[0]
        print((c, d))
        img1 = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
        l1.clear()
        l.clear()
        img = cv2.add(frame, mask)
    else:
        print("List is empty")

    cv2.imshow('frame',img)
    cv2.imwrite("plotlkt.jpg",img1)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
cv2.destroyAllWindows()
cap.release()