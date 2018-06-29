import imutils
import threading
import os
import glob
import numpy as np
import cv2
import csv
import cv2.aruco as aruco
import time
from evaluation import evaluation
#from progress.bar import Bar




def plotcode():
    flag_first = True
    flag_cnt = True
    list_white = []
    img_plot = cv2.imread()
    x_first = 20
    y_first = 266
    img_with_circles = np.zeros((500, 500, 3), np.uint8)

    result_dic = []
    total_frames = 0
    coordinates = [[29.0, 250.0], [38.0, 385.0], [160.97999572753906, 417.5], [114.12354278564453, 338.9093322753906],
                   [88.5, 259.0], [158.53448486328125, 202.6724090576172], [187.5, 38.5],
                   [261.2481384277344, 121.8302230834961], [270.5, 243.0], [291.4565124511719, 422.2826232910156],
                   [387.043701171875, 360.78155517578125], [343.0, 274.5], [362.0, 166.5]]
    for i in coordinates:
        cv2.circle(img_with_circles, (int(i[0]), int(i[1])), 20, (255, 0, 0), thickness=-1)

    # flag_pm = 0
    # cap = cv2.VideoCapture(0)
    # _, frame = cap.read()
    li_pm = []
    pm_list = []
    fps = 30
    cnt_pm = 0

    def aruco_coordinates(file_name):

        ids = None
        flag_contour = 0
        cap = cv2.VideoCapture(file_name)  # Capture videofrom camera
        global fps
        fps = cap.get(cv2.CAP_PROP_FPS)

        while (cap.isOpened() and (flag_contour == 0 or ids == None)):
            ret, frame = cap.read()

            area = (frame.shape)
            frame_area = area[0] * area[1]
            print("Frame Area")
            # print(frame_area)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if (ids == None):
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray = clahe.apply(gray)
                gray_aruco = cv2.GaussianBlur(gray, (5, 5), 0)
                aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
                parameters = aruco.DetectorParameters_create()
                corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_aruco, aruco_dict, parameters=parameters)
                print("id of aruco = ")
                print(ids.flatten())

            if flag_contour == 0:
                # contour filtering part
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
                        # print("Area Percent")
                        # print(cv2.contourArea(c))
                        areapercent = (contour_area / frame_area) * 100
                        # print(areapercent)
                        if areapercent > 25:
                            # print("Arena Found")
                            screenCnt = approx
                            contours = screenCnt
                            flag_contour = 1
                            if flag_contour == 1 and ids != None:
                                break
        print("i am out")
        cap.release()
        return ((ids.flatten()), contours)

    class compute_frame(threading.Thread):

        def __init__(self, filename, num):
            threading.Thread.__init__(self)
            # print("const")
            self.file_name = filename
            self.img = np.zeros((500, 500, 3), np.uint8)
            self.num = num
            img_with_circles = np.zeros((500, 500, 3), np.uint8)
            coordinates = [[29.0, 250.0], [38.0, 385.0], [160.97999572753906, 417.5],
                           [114.12354278564453, 338.9093322753906], [88.5, 259.0],
                           [158.53448486328125, 202.6724090576172], [187.5, 38.5],
                           [261.2481384277344, 121.8302230834961], [270.5, 243.0],
                           [291.4565124511719, 422.2826232910156], [387.043701171875, 360.78155517578125],
                           [343.0, 274.5], [362.0, 166.5]]
            self.circleimage = img_with_circles

        def filter_top_of_robot(self, frame):
            # print("I am in filter_top_of_robot")

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
            if contours.__len__() != 0:
                cnt = contours[0]
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                center = (int(x), int(y))
                # coordinates.append([x,y])
                radius = int(radius)
                cv2.circle(res, center, radius, (0, 255, 0), 2)
                # print("area of circle = ")
                # print((3.14)*(radius*radius))

                if (3.14) * (radius * radius) < 100:
                    # print("small circle")
                    x = 0
                    y = 0
                if flag_first==True:
                    adj_x = x_first - x
                    adj_y = y_first - y
                    global flag_first
                    flag_first = False

                if int(x) != 0 and int(y) != 0:
                    x = x+adj_x
                    y = y+ adj_y
                    if img_plot[int(y),int(x),0]==255:
                        list_white.append(1)
                    else:
                        list_white.append(0)
                    cv2.line(img_with_circles, (int(x), int(y)), (int(x), int(y)), (255, 255, 255), 3)
                    cv2.line(self.img, (int(x), int(y)), (int(x), int(y)), (255, 255, 255), 3)
                else:
                    cv2.line(self.img, (int(x), int(y)), (int(x), int(y)), (255, 255, 255), 3)
                    cv2.line(img_with_circles, (int(x), int(y)), (int(x), int(y)), (255, 255, 255), 3)
            else:
                print("no contour found bro")
            # cv2.imshow("Contour_Filtering",res)
            # cv2.imshow('Plot', self.img)
            # cv2.waitKey(1)

        def warping(self, image, contours):
            # print("I am in warping")

            x1 = contours[0][0][0]
            y1 = contours[0][0][1]
            x2 = contours[1][0][0]
            y2 = contours[1][0][1]
            x3 = contours[2][0][0]
            y3 = contours[2][0][1]
            x4 = contours[3][0][0]
            y4 = contours[3][0][1]
            # print("HII")
            # print((x1, y1))
            # print((x2, y2))
            # print((x3, y3))
            # print((x4, y4))

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
            # cv2.imshow("Warped", dst)
            # cv2.waitKey(1)

            return dst

        def physical_marker(self, frame, flag_pm, li_pm, count_pm, pm_list, start):

            # old_y = 0
            # old_x = 0

            # ret, frame = cap.read()

            # if ret == False:
            #    break
            global cnt_pm
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_red = np.array([92, 103, 191])
            upper_red = np.array([111, 195, 255])
            mask = cv2.inRange(hsv, lower_red, upper_red)
            res = cv2.bitwise_and(frame, frame, mask=mask)
            if flag_pm == 0:

                kernel = np.ones((5, 5), np.uint8)

                gray = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
                gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
                gray = cv2.bilateralFilter(gray, 11, 17, 17)
                # edged = cv2.Canny(gray, 30, 200)

                gray = cv2.erode(gray, kernel, iterations=1)
                gray = cv2.dilate(gray, kernel, iterations=5)

                (_, contours, _) = cv2.findContours(gray.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contours = sorted(contours, key=cv2.contourArea, reverse=True)[:4]
                for cnt in contours:
                    # print("cnt = ")
                    # print(cnt)
                    (x, y), radius = cv2.minEnclosingCircle(cnt)
                    center = (int(x), int(y))
                    # coordinates.append([x,y])
                    radius = int(radius)
                    # cv2.circle(res, center, radius, (0, 255, 0), 2)
                    li_pm.append(center)
                    # old_x=x
                    # old_y=y

                # print(li)

                if li_pm.__len__() == 4:
                    print(li_pm.__len__())
                    # print(index_list)
                    flag_pm = 1
                else:
                    print("i am here")
                    li_pm.clear()

            for c in li_pm:
                x = int(res[c[1], c[0]][0])
                y = int(res[c[1], c[0]][1])
                z = int(res[c[1], c[0]][2])

                # print(li.index(c))

                if (x + y + z) == 0:
                    cnt_pm += 1
                    print("in if")
                    li_pm.pop(li_pm.index(c))
                    if li_pm.__len__() % 2 != 0:
                        flag_cnt = False
                        # start = time.time()
                    else:
                        frame_value = cnt_pm
                        # print("time stops")
                        # end = time.time()
                        # print("Time taken = ")
                        print(frame_value)
                        time_taken = frame_value / fps
                        pm_list.append(time_taken)
                        if li_pm.__len__() == 0:
                            break
                    if flag_cnt == False:
                        cnt_pm+=1



            return (flag_pm, li_pm, count_pm)  # ,pm_list,start)

        def deleteframes(self, team_id, file_name, contours, flag=True):
            count = 0
            # name = "team_id_" + str(team_id[0]) + ".png"
            if not os.path.exists(os.path.join(os.getcwd(), "plot")):
                os.makedirs(os.path.join(os.getcwd(), "plot"))
            if not os.path.exists(os.path.join(os.getcwd(), "plot", "teamid_" + str(self.num))):
                os.makedirs(os.path.join(os.getcwd(), "plot", "teamid_" + str(self.num)))
            # name = "team_id_" + str(self.num) + ".png"

            # print(name)
            cap = cv2.VideoCapture(file_name)

            flag_pm = 0
            li_pm = []
            count_pm = 0
            pm_list = []
            start = 0

            while (cap.isOpened()):

                # print("I am in deleteframes")
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

                if ret == True and (tlx, tly, trix, triy, blx, bly, brx, bry) != (0, 0, 0, 0, 0, 0, 0, 0):
                    flag = True

                    warped_frame = self.warping(image, contours)

                    self.filter_top_of_robot(warped_frame)

                elif (tlx, tly, trix, triy, blx, bly, brx, bry) == (0, 0, 0, 0, 0, 0, 0, 0):
                    if flag == True:
                        count += 1
                        print("count = " + str(count))
                        flag = False

                # cv2.imshow("Original", image)
                # cv2.waitKey(1)

            cap.release()
            cv2.destroyAllWindows()
            listlen = list_white.__len__()
            list_ones = list_white.count(1)
            followaccuracy = (list_ones/listlen)*100
            print("ACUURACY MEASURE",followaccuracy)
            name = os.path.join(
                os.path.join(os.getcwd(), "plot", "teamid_" + str(self.num), "plot_" + str(count) + "_" + ".png"))
            name2 = os.path.join(os.path.join(os.getcwd(), "plot", "teamid_" + str(self.num), "plot_circle.png"))
            # print(name)
            # print(name2)
            # print("i am out of delete_frames")
            cv2.imwrite(name, self.img)
            cv2.imwrite(name2, img_with_circles)


        ##############################################################################################################################################################
        ##############################################################################################################################################################
        def run(self):
            print("in run")
            csv_file_name = "C:\\Users\\Saim Shaikh\\Desktop\\Eysip\\Results\\results.csv"
            global result_dic
            # start = time.time()

            (team_id, coordinates) = aruco_coordinates(self.file_name)
            self.deleteframes(team_id, self.file_name, coordinates)
            print("finished" + str(self.num))


    if __name__ == '__main__':
        path = 'C:\\Users\\Saim Shaikh\\Desktop\\Eysip\\videos\\'
        # compute_frame th2
        files = glob.glob(path + '*.mov')

        for i in files:
            th = compute_frame(i,i.index(i))
            th.start()






plotcode()

