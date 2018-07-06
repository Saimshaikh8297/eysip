import imutils
import threading
import os
import glob
import numpy as np
import cv2
import csv
import cv2.aruco as aruco
import ast
import time
import sys
import pandas
#from evaluation import evaluation

'''
Scope : GLOBAL
Function : Read the data from the csv file written by the standard file containing the results and path of the images
'''
#magenta #neon green #green #blue
img_plot = [] # Reading the image with thick line
img_with_circles = [] # Reading the image with circles

hsv_values = [[[124,112,171],[148,193,255]],[[32,102,202],[43,163,255]],[[46,38,163],[67,170,255]],[[92,103,191],[111,195,255]]]
perfect_values= [] # This list will be formed once in the program. It will store the values from the csv file of perfect run
path_to_circle_image = "" #parsing through the list and giving the path of the image with circles
path_to_thick_line = ""  #parsing through the list and giving the path of the image with thick line
# perfect_values[0][3] = [] #parsing through the list and giving the first coordinate of perfect trajectory
x_first = 0
y_first = 0 # x_first & y_first store integer value of the x & y coordinates of the first pixel of perfect trajectory
result_dic = [] # Each result's dictionary will be appended in this list
th =[] # The threads formed will be appended in this list


def mainplot(csvpath,folderpath,hsvtop,hsvphysical):
    global perfect_values,path_to_circle_image,path_to_thick_line,x_first,y_first,img_plot,img_with_circles
    with open(csvpath, 'r') as csvfile:  # Opening the csv file
        # creating a csv reader object
        csvreader = csv.reader(csvfile)

        # extracting field names through first row
        fields = next(csvreader)

        # extracting each data row one by one
        for row in csvreader:
            perfect_values.append(row)

    path_to_circle_image = perfect_values[0][1]  # parsing through the list and giving the path of the image with circles
    path_to_thick_line = perfect_values[0][2]  # parsing through the list and giving the path of the image with thick line
    perfect_values[0][3] = ast.literal_eval(perfect_values[0][3])  # parsing through the list and giving the first coordinate of perfect trajectory
    (x_first, y_first) = perfect_values[0][3][0]  # x_first & y_first store integer value of the x & y coor
    img_plot = cv2.imread(path_to_thick_line)  # Reading the image with thick line
    img_with_circles = cv2.imread(path_to_circle_image)  # Reading the image with circles
    plotcode(folderpath,hsvtop,hsvphysical)
    csv_file_name = os.path.join(os.getcwd(), 'Results', 'intermediate_results.csv')

    fields = ['Team_ID','Plot Path','Circle Path', 'Handling Count', "Physical Marker 1 Time", "Physical Marker 2 Time", "Follow Accuracy"]
    with open(csv_file_name, 'w') as csvfile:
        # creating a csv dict writer object
        writer = csv.DictWriter(csvfile, fieldnames=fields)

        # writing headers (field names)
        writer.writeheader()

        # writing data rows
        #print(result_dic)
        writer.writerows(result_dic)





#print(path_to_thick_line)

'''
Scope : GLOBAL
Function: To be used as reference lists in which items can be appended throughout the execution
'''


'''
Function Name : Plotcode
Parameters : None
Usage : This function contains the entire code which will firstly initiate the threads
'''
def plotcode(folderpath,hsvtop,hsvphysical):
    print("in plotcode")



    '''
    Function name : aruco_coorinates
    Parameters: file_name(Path of the video file)
    Usage : This function will be called first as soon as thread is initialised.
            It will return the ID of the Team and the 4 coordinates of the Arena
    '''
    def aruco_coordinates(file_name):

        ids = None # initialising the ID as none
        flag_contour = 0 # flag_contour variable is set at 0 until the arena is found in the frame. Then it is set to 1
        cap = cv2.VideoCapture(file_name)  # Capture video from camera

        '''
        This while loop will continue to run until the ID of the team AND the Arena is found.
        If either of the thing is not found it will return False.
        '''

        while (cap.isOpened() and (flag_contour == 0 or ids == None)):
            print("Hi")
            ret, frame = cap.read()

            area = (frame.shape) # Area of the frame
            frame_area = area[0] * area[1] # Calaculating the area of the frame for comparison later
            print("Frame Area")
            # print(frame_area)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)# Converting the frame to Grayscale for finding Aruco
            '''
            This if will be executed if the ID of the Aruco has not been found.
            The variable ids will store the ID.
            '''
            if (ids == None):
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray = clahe.apply(gray)
                gray_aruco = cv2.GaussianBlur(gray, (5, 5), 0)
                aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
                parameters = aruco.DetectorParameters_create()
                corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_aruco, aruco_dict, parameters=parameters)
                # print("id of aruco = ")
                # print(ids.flatten())
            '''
            This if will be executed until the Arena has been found from the frame
            To find the Arena, following steps are taking place:
            '''
            if flag_contour == 0:
                # contour filtering part
                blurred = cv2.bilateralFilter(gray, 11, 17, 17) # Blurring the frame
                kernel = np.ones((5, 5), np.uint8) # Making a 5x5 Kernel
                blurredopen = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel) # Morphological Opening
                blurredopen = cv2.morphologyEx(blurredopen, cv2.MORPH_OPEN, kernel) # Morphological Closing
                blurredclose = cv2.morphologyEx(blurredopen, cv2.MORPH_CLOSE, kernel) # Morphological Closing
                edged = cv2.Canny(blurredclose, 30, 200) # Canny Edge Detection
                cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Finding contours from the edged frame
                # cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
                cnts = cnts[0] if imutils.is_cv2() else cnts[1]
                cntsSorted = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)[:1] # Putting the contours in the order of decreasing Area
                for c in cntsSorted:
                    # approximate the contour
                    peri = cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, 0.01 * peri, True)
                    # if our approximated contour has four points, then
                    # we can assume that we have found our screen
                    if len(approx) == 4: # Checking if the Contour found has 4 corners
                        contour_area = (cv2.contourArea(c)) # Finding contour Area
                        # print("Area Percent")
                        # print(cv2.contourArea(c))
                        areapercent = (contour_area / frame_area) * 100 # As the arena will occupy Maximum Area of the Frame, we use it to find it
                        # print(areapercent)
                        if areapercent > 25: # If Contours's Area > 25% of the Total Area of the Frame, Then it is the Arena
                            # print("Arena Found")
                            screenCnt = approx
                            contours = screenCnt
                            flag_contour = 1 # Setting flag_contour to 1 as the Arena has been found
                            if flag_contour == 1 and ids != None:
                                break
        print("i am out")
        print(ids)
        cap.release()
        return ((ids.flatten()[0]), contours)

    '''
    Making a threading class.
    This class will perform threading.
    The constructor will initialise each thread 
    '''
    # class evaluating_thread(threading.Thread):
    #
    #     def __init__(self):
    #         threading.Thread.__init__(self)
    #         print("in eval thread")



    class compute_frame(threading.Thread):

        def __init__(self, filename, num):
            threading.Thread.__init__(self)
            print("const")
            self.file_name = filename #file_name is passed to the thread
            self.img = np.zeros((500, 500, 3), np.uint8) # Black image where the trajectory will be plotted
            self.num = num # File_number is passed to thread for debugging purpose
            self.circleimage = img_with_circles.copy() # Image with circle for programmatic evaluation
            self.adj_x = 0 #this variable is used to adjust the offset while evaluating the trajectory
            self.adj_y = 0 #this variable is used to adjust the offset while evaluating the trajectory
            self.flag_first = True #This flag will be used to get first coordinates of the trajectory from thie video
            self.flag_cnt = True #T
            self.list_white = [] # This list will store the number of pixels plotted in and out of the reference trajectory for evaluation
            self.pm_framecounts = [] # This list will contain the number of frames between 2 physical markers
            self.li_pm = [] # This list will contain the number of color markers on the arena
            self.pm_list = [] #
            self.cnt_pm = 0 # This variable will be used to count the number of frames between 2 physical markers
            self.total_frames = 0 #
            self.fps = 30 #FPSof the video
            self.flag_pm=0 #This flag will be set only if all of the number of color markers on the arena have been found




        '''
        Function Name : filter_top_of_robot
        Parameters: frame
        Return : None
        Usage : This function will color filter the frame to find the color marker placed on the robot and plot a point on the position of its 
        centroid on 3 images: the plane black image(For feature matching and HOG) and the circular image and the thick image(For Programmatic evaluation)
        This function works in the following way:
        '''
        def filter_top_of_robot(self, frame):
            # global adj_x
            # global  adj_y

            # print("I am in filter_top_of_robot")

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #Converting the frame to HSV
            #print(hsv_values[hsvtop][0])
            lower_red = np.array(hsv_values[hsvtop][0]) #Color range for Magenta
            upper_red = np.array(hsv_values[hsvtop][1])

            mask = cv2.inRange(hsv, lower_red, upper_red) # Applying a Mask to filter out color
            res = cv2.bitwise_and(frame, frame, mask=mask) # Doing bitwise and to subtract all other colors

            gray = cv2.cvtColor(res, cv2.COLOR_HSV2BGR) # Converting HSV to BGR
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY) # Converting BGR to Gray
            gray = cv2.bilateralFilter(gray, 11, 17, 17) # Applying Bilateral Filter to reduce noise
            edged = cv2.Canny(gray, 30, 200) # Applying Canny Edge Detection

            (_, contours, _) = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) # FInding contours
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
            # global flag_first
            if contours.__len__() != 0:
                cnt = contours[0]
                (x, y), radius = cv2.minEnclosingCircle(cnt) # Making Minimum Enclosing Circle around the contour to get the coordinates of the centre
                center = (int(x), int(y))
                # coordinates.append([x,y])
                radius = int(radius)
                cv2.circle(res, center, radius, (0, 255, 0), 2)
                # print("area of circle = ")
                # print((3.14)*(radius*radius))

                if (3.14) * (radius * radius) < 700: # This will filter out small contours which are found
                    # print("small circle")
                    x = 0
                    y = 0

                if self.flag_first==True and (x,y)!=(0,0): # This if will be executed if it is the first pixel in the trajectory
                    #print("adj")
                    #print(x,y)
                    #print(x_first,y_first)
                    self.adj_x = x_first - x
                    self.adj_y = y_first - y
                    # print(adj_x,adj_y)
                    self.flag_first = False

                if int(x) != 0 and int(y) != 0:
                    x = x + self.adj_x
                    y = y + self.adj_y
                    #print(y,x)
                    if img_plot[int(y),int(x),0]==255: # Check if the pixel is plotted on White Foreground or Black Background
                        self.list_white.append(1)
                    else:
                        self.list_white.append(0)
                    cv2.line(self.circleimage, (int(x), int(y)), (int(x), int(y)), (255, 255, 255), 3)
                    cv2.line(self.img, (int(x), int(y)), (int(x), int(y)), (255, 255, 255), 1)
                else:
                    cv2.line(self.img, (int(x), int(y)), (int(x), int(y)), (255, 255, 255), 3)
                    cv2.line(self.circleimage, (int(x), int(y)), (int(x), int(y)), (255, 255, 255), 1)
            else:
                print("no contour found bro")
            # cv2.imshow("Contour_Filtering",res)
            # cv2.imshow('Plot', self.img)
            # cv2.waitKey(1)

        '''
        Function Name: warping
        Parameters: Frame, Coordinates
        Returns : The warped Frame which can be used further for color filtering
        '''
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

            s1 = x1 + y1
            s2 = x2 + y2
            s3 = x3 + y3
            s4 = x4 + y4

            t = max(s1, s2, s3, s4)
            if t == s1:
                x2_main = x1
                y2_main = y1
                x1 = 0
                y1 = 0

            elif t == s2:
                x2_main = x2
                y2_main = y2
                x2 = 0
                y2 = 0

            elif t == s3:
                x2_main = x3
                y2_main = y3
                x3 = 0
                y3 = 0

            else:
                x2_main = x4
                y2_main = y4
                x4 = 0
                y4 = 0

            #print(x2_main, y2_main)

            t = min(s1, s2, s3, s4)
            if t == s1:
                x4_main = x1
                y4_main = y1
                x1 = 0
                y1 = 0

            elif t == s2:
                x4_main = x2
                y4_main = y2
                x2 = 0
                y2 = 0

            elif t == s3:
                x4_main = x3
                y4_main = y3
                x3 = 0
                y3 = 0

            else:
                x4_main = x4
                y4_main = y4
                x4 = 0
                y4 = 0
            #print(x4_main, y4_main)

            t = max(x1, x2, x3, x4)
            x3_main = t
            index_min = np.argmax([x1, x2, x3, x4])
            if index_min == 0:
                x1 = 0
            elif index_min == 1:
                x2 = 0
            elif index_min == 2:
                x3 = 0
            else:
                x4 = 0

            t = max(x1, x2, x3, x4)
            x1_main = t

            t = max(y1, y2, y3, y4)
            y1_main = t
            index_min = np.argmax([y1, y2, y3, y4])
            if index_min == 0:
                y1 = 0
            elif index_min == 1:
                y2 = 0
            elif index_min == 2:
                y3 = 0
            else:
                y4 = 0

            t = max(y1, y2, y3, y4)
            y3_main = t

            mask = np.zeros(image.shape, dtype=np.uint8)
            roi_corners = np.array([[(x1_main, y1_main), (x2_main, y2_main), (x3_main, y3_main), (x4_main, y4_main)]], dtype=np.int32)
            channel_count = image.shape[2]
            ignore_mask_color = (255,) * channel_count
            cv2.fillPoly(mask, roi_corners, ignore_mask_color)
            masked_image = cv2.bitwise_and(image, mask)

            pts1 = np.float32([(x3_main, y3_main), (x2_main, y2_main), (x4_main, y4_main), (x1_main, y1_main)])
            pts2 = np.float32([[0, 0], [500, 0], [0, 500], [500, 500]])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            dst = cv2.warpPerspective(masked_image, M, (500, 500))
            # cv2.imshow("Warped", dst)
            # cv2.waitKey(1)

            return dst

        '''
        Function Name: physical_marker
        Parameters: frame
        Returns : None
        '''
        def physical_marker(self, frame):

            # global cnt_pm
            # global pm_framecounts
            # global flag_cnt
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_red = np.array(hsv_values[hsvphysical][0])
            upper_red = np.array(hsv_values[hsvphysical][1])
            mask = cv2.inRange(hsv, lower_red, upper_red)
            res = cv2.bitwise_and(frame, frame, mask=mask)
            if self.flag_pm == 0:

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
                    self.li_pm.append(center)
                    # old_x=x
                    # old_y=y

                # print(li)

                if self.li_pm.__len__() == 4:
                    print(self.li_pm.__len__())
                    # print(index_list)
                    self.flag_pm = 1
                else:
                    #print("i am here")
                    self.li_pm.clear()

            for c in self.li_pm:
                #print(self.li_pm.__len__())
                x = int(res[c[1], c[0]][0])
                y = int(res[c[1], c[0]][1])
                z = int(res[c[1], c[0]][2])

                # print(li.index(c))

                if (x + y + z) == 0:
                    self.cnt_pm += 1
                    print("in if")
                    self.li_pm.pop(self.li_pm.index(c))
                    if self.li_pm.__len__() % 2 != 0:
                        self.flag_cnt = False
                        # start = time.time()
                    else:
                        self.flag_cnt = True
                        # frame_value = self.cnt_pm
                        # print("time stops")
                        # end = time.time()
                        # print("Time taken = ")
                        # print(frame_value)
                        # time_taken = frame_value / fps
                        # pm_list.append(time_taken)
                        if self.li_pm.__len__()==2:
                            print("physical1")
                            self.pm_framecounts.append(self.cnt_pm)
                            self.cnt_pm=0
                        if self.li_pm.__len__() == 0:
                            print("physical2")
                            self.pm_framecounts.append(self.cnt_pm)
                            break
            if self.flag_cnt == False:
                self.cnt_pm += 1


        def deleteframes(self, team_id, file_name, contours, flag=True):
            count = 0
            plot_path = "team_id_" + str(self.num) + ".png"
            circle_path = "team_id_" + str(self.num) + "circle.png"

            # print(name)
            cap = cv2.VideoCapture(file_name)
            self.fps = cap.get(cv2.CAP_PROP_FPS)

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


                    if(hsvphysical!=None):
                        self.physical_marker(image)

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
            listlen = self.list_white.__len__()
            list_ones = self.list_white.count(1)
            followaccuracy = (list_ones/listlen)*100
            print("ACUURACY MEASURE",followaccuracy)

            # print(name)
            # print(name2)
            # print("i am out of delete_frames")
            # print(self.pm_framecounts)
            # print(self.cnt_pm)
            # print(fps)
            if(hsvphysical==None):
                tmarker1 = "N/A"
                tmarker2 = "N/A"
            else:
                tmarker1 = self.pm_framecounts[0] / self.fps
                tmarker2 = self.pm_framecounts[1] / self.fps
                print("Physical mARKER 1 Point " + str(tmarker1))
                print("Physical mARKER 2 Point " + str(tmarker2))


            result = {'Team_ID': team_id, 'Plot Path':plot_path , 'Circle Path':circle_path ,'Handling Count': count,
                      "Physical Marker 1 Time":tmarker1, "Physical Marker 2 Time":tmarker2,
                      "Follow Accuracy":followaccuracy}
            global result_dic
            result_dic.append(result.copy())
            # print(result)
            # print(result_dic)
            cv2.imwrite(plot_path, self.img)
            cv2.imwrite(circle_path, self.circleimage)


        ##############################################################################################################################################################
        ##############################################################################################################################################################
        def run(self):
            print("in run")
            (team_id, coordinates) = aruco_coordinates(self.file_name)
            print("team_id" + str(team_id))
            self.deleteframes(team_id, self.file_name, coordinates)
            print("finished" + str(self.num))



    path = folderpath
    print(folderpath)



    files = glob.glob(path + '*.mov' or '*.mp4')
    print(files)
    index=0
    #print("Length"+str(files.__len__()))
    for i in range(0,files.__len__()):


        if i<3:
            #print("hohi")
            th.append(compute_frame(files[i],i))
            th[i].start()
            index=i

    while True:
        time.sleep(2)
        for i in range(1):
            if not(th[i].is_alive()):
                print("thread"+str(i)+"is closed")
                index+=1
                if index<files.__len__():
                  th[i] = compute_frame(files[index],index)
                else:
                    print("All files are in thread")
                    #print(threading.active_count())
                    if threading.active_count()==1:
                        return



#mainplot(os.path.join(os.getcwd(), "Results", "results_perfect.csv"),"/Users/siddharth/Desktop/EYSIP/NEW VIDS & RESULTS/videos/",0,3)