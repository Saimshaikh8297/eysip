import cv2
import numpy as np
import os
import csv
import ast

'''

* Project:           e-Yantra Automatic Evaluation of Videos
* Author List:       Saim Shaikh, Siddharth Aggarwal
* Filename:          evaluation_modular.py
* Functions:         eval, programmatic, feature_matching, HOG_correlation
* Global Variables:  perfect_values,path_to_perfect_image,feature_list,feature_list,coordinates
                     img_perfect,csv_file_name,result_dic

'''

perfect_values= []
team_ids = []
handling_counts = []
pms =[]
faccs = []

with open("C:\\Users\\Saim Shaikh\\Desktop\\Eysip\\Results\\perfect_results.csv",'r') as csvfile:

    # creating a csv reader object
    csvreader = csv.reader(csvfile)
    # extracting field names through first row
    fields = next(csvreader)
    print(csvreader)
    # extracting each data row one by one
    for row in csvreader:
        # print(row)
        perfect_values.append(row)



# print(perfect_values)
with open(os.path.join(os.getcwd(),"Results","results.csv"), 'r') as csvfile2:
    # creating a csv reader object
    csvreader2 = csv.reader(csvfile2)

    # extracting field names through first row
    fields2 = next(csvreader2)

    # extracting each data row one by one
    for row in csvreader2:
        team_ids.append(row[0])
        handling_counts.append(row[1])
        pms.append([row[2],row[3]])
        faccs.append(row[4])

# print(perfect_values)
path_to_perfect_image = perfect_values[0][1] #extracting perfect trajectory for evaluation
perfect_values[0][4] = ast.literal_eval(perfect_values[0][4]) #akaze values of the perfect trajectory
feature_list = perfect_values[0][4] #akaze values of the perfect trajectory
coordinates = ast.literal_eval(perfect_values[0][3]) #co-ordinates of checkpoints for programmatic analysis
print(coordinates)
img_perfect = cv2.imread(path_to_perfect_image)
csv_file_name = os.path.join(os.getcwd(),'Results','results_final.csv') #creating csv file to store the results
result_dic = []



def evaluation():

    #extracting the trajectory ,team id and the aruco count from the folders
    for i in os.listdir(os.path.join(os.getcwd(), 'plot')):
        team_id = team_ids[i.index(i)]
        pm1 = pms[i.index(i)][0]
        pm2 = pms[i.index(i)][1]
        facc = faccs[i.index(i)]


        print("Team ID: "+team_id)
        for j in os.listdir(os.path.join(os.getcwd(), 'plot', i)):
            if (j.find('circle') != -1):
                img_circle=cv2.imread(os.path.join(os.getcwd(), 'plot', i,j))
                checkpoint_result = programmatic(img_circle) #saving the programmatic analysis results
                result = {'Team_ID': team_id, 'Handling Count': count,"PM1":pm1,"PM2":pm2,"Follow_Accuracy":facc,
                          'Programmatic_Score': checkpoint_result, 'Feature_Matching_Avg_Score': feature_result,
                          'HOG_Score': hog_result}
                result_dic.append(result.copy()) #appending the programmatic analysis reult to the csv files

            else:
                count = handling_counts[i.index(i)]
                print("Handling: "+count)
                img_plot = cv2.imread(os.path.join(os.getcwd(), 'plot', i,j))
                feature_result = 0
                feature_result = feature_match(img_plot) #saving the feature matching result
                hog_result = HOG_correlation(img_plot)  #saving the HOG result

    fields = ['Team_ID', 'Handling Count','PM1','PM2','Follow_Accuracy', 'Programmatic_Score', 'Feature_Matching_Avg_Score', 'HOG_Score',
              ]
    with open(csv_file_name, 'w') as csvfile:
       # creating a csv dict writer object
       writer = csv.DictWriter(csvfile, fieldnames=fields)

       # writing headers (field names)
       writer.writeheader()

       # writing data rows
       writer.writerows(result_dic)

##################################### PROGRAMMATIC ANALYSIS OF CHECK-POINTS #########################################

'''
Function Name:     programmatic
Input :            plot with the programatic circles
Output :           result of programmatic analysis in percentage

Logic :            In this function we accept a plot with the programmatic circles
                   plotted according to the checkpoint coordinates, we extract the
                   roi of these circles and check whether the trajectory is passing
                   through the circles.Based on this we calculate the score.     
'''
def programmatic(img_circle):

    # programmatic checkpoints
    circle_radius = 8 #raduis of the programmatic circle
    check_list = [] #list of circles passed by the trajectory
    check_counter = 0 #count of checkpoints crossed

    #iterating through the co-ordinates
    for i in coordinates:
        print(i)
        a = int(i[0])
        b = int(i[1])

        #extracting the roi of the plotted circle
        roi = img_circle[b - (3 * circle_radius): b + (3 * circle_radius),
              a - (3 * circle_radius): a + (3 * circle_radius)]
        roi = roi.reshape(int(roi.size / 3), 3)

        #checking whether the trajectory is passing through the roi
        if [255, 255, 255] in roi.tolist():
            check_list.append(1)
            check_counter += 1

        else:
            check_list.append(0)

    #checking result
    check_result = ((check_counter / check_list.__len__()) * 100)

    print("Programmatic Analysis Result = ")
    print(check_result)
    return check_result
    # load the image and convert it to grayscale



'''
Function Name:     feature_match
Input :            plot with the trajectory
Output :           feature matching results

Logic :           In this function various feature matching algorithms like
                  AKAZE BRISK are used to extract and match features between 
                  the perfect and imperfect trajectories. Based on the number
                  of correct matches a score is calculated  
'''
####################################### ANALYSIS USING FEATURE MATCHING ##################################################
def feature_match(img_akaze):
    # load the image and convert it to grayscale
    gray1 = cv2.cvtColor(img_perfect, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img_akaze, cv2.COLOR_BGR2GRAY)

    # initialize the AKAZE,BRISK descriptor, then detect keypoints and extract
    # local invariant descriptors from the image
    sift = cv2.xfeatures2d.SIFT_create()
    surf = cv2.xfeatures2d.SURF_create()
    akaze = cv2.AKAZE_create()
    brisk = cv2.BRISK_create()
    orb = cv2.ORB_create()

    #compute the descriptors and keypoints using AKAZE BRISK ORB SIFT and SURF
    (akazekps1, akazedescs1) = akaze.detectAndCompute(gray1, None)
    (akazekps2, akazedescs2) = akaze.detectAndCompute(gray2, None)
    (siftkps1, siftdescs1) = sift.detectAndCompute(gray1, None)
    (siftkps2, siftdescs2) = sift.detectAndCompute(gray2, None)
    (surfkps1, surfdescs1) = surf.detectAndCompute(gray1, None)
    (surfkps2, surfdescs2) = surf.detectAndCompute(gray2, None)
    (briskkps1, briskdescs1) = brisk.detectAndCompute(gray1, None)
    (briskkps2, briskdescs2) = brisk.detectAndCompute(gray2, None)
    (orbkps1, orbdescs1) = orb.detectAndCompute(gray1, None)
    (orbkps2, orbdescs2) = orb.detectAndCompute(gray2, None)

    # Match the fezatures using the Brute Force Matcher
    bfakaze = cv2.BFMatcher(cv2.NORM_HAMMING)
    bf = cv2.BFMatcher(cv2.NORM_L2)

    #Refine the Brute Force Matches using the KNN Matcher
    akazematches = bfakaze.knnMatch(akazedescs1, akazedescs2, k=2)
    siftmatches = bf.knnMatch(siftdescs1, siftdescs2, k=2)
    surfmatches = bf.knnMatch(surfdescs1, surfdescs2, k=2)
    briskmatches = bf.knnMatch(briskdescs1, briskdescs2, k=2)
    orbmatches = bf.knnMatch(orbdescs1, orbdescs2, k=2)

    # Apply ratio test on AKAZE matches
    goodakaze = []
    for m, n in akazematches:
        if m.distance < 0.9 * n.distance:
            goodakaze.append([m])
    goodakaze = np.asarray(goodakaze)
    print(feature_list)
    print(goodakaze.shape[0])
    #calculate the Akaze core using the number of good matches
    similarity_akaze = (goodakaze.shape[0]/feature_list[0][0])*100

    # Apply ratio test on SIFT matches
    goodsift = []
    for m, n in siftmatches:
        if m.distance < 0.9 * n.distance:
            goodsift.append([m])
    im3sift = cv2.drawMatchesKnn(img_perfect, siftkps1, img_akaze, siftkps2, goodsift[:], None, flags=2)
    goodsift = np.asarray(goodsift)
    similarity_sift = (goodsift.shape[0] / feature_list[1][0]) * 100

    # Apply ratio test on SURF matches
    goodsurf = []
    for m, n in surfmatches:
        if m.distance < 0.9 * n.distance:
            goodsurf.append([m])
    goodsurf = np.asarray(goodsurf)
    similarity_surf = (goodsurf.shape[0] / feature_list[2][0]) * 100

    # Apply ratio test on ORB matches
    goodorb = []
    for m, n in orbmatches:
        if m.distance < 0.9 * n.distance:
            goodorb.append([m])
    goodorb = np.asarray(goodorb)
    similarity_orb = (goodorb.shape[0] / feature_list[3][0]) * 100

    # Apply ratio test on BRISK matches
    goodbrisk = []
    for m, n in briskmatches:
        if m.distance < 0.9 * n.distance:
            goodbrisk.append([m])
    goodbrisk = np.asarray(goodbrisk)

    #Calculating the Similarity using the BRISK algorithm
    similarity_brisk = (goodbrisk.shape[0] / feature_list[4][0]) * 100
    features_result = (similarity_akaze+similarity_brisk+similarity_orb+similarity_sift+similarity_surf)/5

   #calculating overall similarity by aggregating the results of various feature actching algorithms
    print("Overall similarity using features: ")
    print(features_result)
    return features_result



    ######################################### HOG CORRELATION ###############################################
def HOG_correlation(img_akaze):
    bin_n = 16

    img = img_akaze
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)

    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)

    mag, ang = cv2.cartToPolar(gx, gy)

    # quantizing binvalues in (0...16)

    bins = np.int32(bin_n * ang / (2 * np.pi))

    # Divide to 4 sub-squares

    bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], bins[10:, 10:]

    mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]

    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]

    hist1 = np.hstack(hists)

    img = img_perfect
    rows, cols, _ = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 0, 1)
    img = cv2.warpAffine(img, M, (cols, rows))

    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)

    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)

    mag, ang = cv2.cartToPolar(gx, gy)

    # quantizing binvalues in (0...16)

    bins = np.int32(bin_n * ang / (2 * np.pi))

    # Divide to 4 sub-squares

    bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], bins[10:, 10:]

    mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]

    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]

    hist2 = np.hstack(hists)

    # print(np.corrcoef(hist1,hist2))

    hog_result = ((np.corrcoef(hist1, hist2)[0][1]) * 100)
    print("HOG CORRELATION RESULT = ")
    print(hog_result)
    return hog_result

evaluation()