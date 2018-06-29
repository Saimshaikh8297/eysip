import cv2
import numpy as np
import os
import csv


img_perfect = cv2.imread("team_id_2_comparison.png")

csv_file_name = os.path.join(os.getcwd(),'Results','results.csv')
coordinates = [[29.5, 250.0], [38.0, 385.0], [160.97999572753906, 417.5],
               [114.12354278564453, 338.9093322753906], [88.5, 259.0], [158.53448486328125, 202.6724090576172],
               [187.5, 38.5], [261.2481384277344, 121.8302230834961], [270.5, 243.0],
               [291.4565124511719, 422.2826232910156], [387.043701171875, 360.78155517578125], [343.0, 274.5],
               [362.0, 166.5]]

feature_list = [636, 395, 1046, 500, 1605]
result_dic = []


def eval():

    for i in os.listdir(os.path.join(os.getcwd(), 'plot')):
        team_id = i.split("_")[1]
        print("Team ID: "+team_id)
        for j in os.listdir(os.path.join(os.getcwd(), 'plot', i)):
            # print(j)
            if (j.find('circle') != -1):
                img_circle=cv2.imread(os.path.join(os.getcwd(), 'plot', i,j))
                checkpoint_result = programmatic(img_circle)
                result = {'Team_ID': team_id, 'Handling Count': count,
                          'Programmatic_Score': checkpoint_result, 'Feature_Matching_Avg_Score': feature_result,
                          'HOG_Score': hog_result}  # ,'PM 1':pm_list[0],'PM 2':pm_list[1]}
                result_dic.append(result.copy())

            else:
                count = j.split('_')[1]
                print("Handling: "+count)
                img_plot = cv2.imread(os.path.join(os.getcwd(), 'plot', i,j))
                feature_result = feature_match(img_plot)
                hog_result = HOG_correlation(img_plot)

    fields = ['Team_ID', 'Handling Count', 'Programmatic_Score', 'Feature_Matching_Avg_Score', 'HOG_Score',
              'PM 1', 'PM 2']
    with open(csv_file_name, 'w') as csvfile:
       # creating a csv dict writer object
       writer = csv.DictWriter(csvfile, fieldnames=fields)

       # writing headers (field names)
       writer.writeheader()

       # writing data rows
       writer.writerows(result_dic)




##################################### PROGRAMMATIC ANALYSIS OF CHECK-POINTS #########################################

def programmatic(img_circle):




    # programmatic checkpoints
    circle_radius = 8
    check_list = []
    check_counter = 0

    for i in coordinates:
        i[0] = int(i[0])
        i[1] = int(i[1])

        roi = img_circle[i[1] - (3 * circle_radius): i[1] + (3 * circle_radius),
              i[0] - (3 * circle_radius): i[0] + (3 * circle_radius)]
        roi = roi.reshape(int(roi.size / 3), 3)

        if [255, 255, 255] in roi.tolist():
            check_list.append(1)
            check_counter += 1

        else:
            check_list.append(0)

    check_result = ((check_counter / check_list.__len__()) * 100)

    print("Programmatic Analysis Result = ")
    print(check_result)
    return check_result
    # load the image and convert it to grayscale



####################################### ANALYSIS USING FEATURE MATCHING ##################################################
def feature_match(img_akaze):
    # load the image and convert it to grayscale
    gray1 = cv2.cvtColor(img_perfect, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img_akaze, cv2.COLOR_BGR2GRAY)

    # initialize the AKAZE descriptor, then detect keypoints and extract
    # local invariant descriptors from the image
    sift = cv2.xfeatures2d.SIFT_create()
    surf = cv2.xfeatures2d.SURF_create()
    akaze = cv2.AKAZE_create()
    brisk = cv2.BRISK_create()
    orb = cv2.ORB_create()

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

    #print("No of KeyPoints:")
    #print("akazekeypoints1: {}, akazedescriptors1: {}".format(len(akazekps1), akazedescs1.shape))
    #print("akazekeypoints2: {}, akazedescriptors2: {}".format(len(akazekps2), akazedescs2.shape))
    #print("siftkeypoints1: {}, siftdescriptors1: {}".format(len(siftkps1), siftdescs1.shape))
    #print("siftkeypoints2: {}, siftdescriptors2: {}".format(len(siftkps2), siftdescs2.shape))
    #print("surfkeypoints1: {}, surfdescriptors1: {}".format(len(surfkps1), surfdescs1.shape))
    #print("surfkeypoints2: {}, surfdescriptors2: {}".format(len(surfkps2), surfdescs2.shape))
    #print("briskkeypoints1: {}, briskdescriptors1: {}".format(len(briskkps1), briskdescs1.shape))
    #print("briskkeypoints2: {}, briskdescriptors2: {}".format(len(briskkps2), briskdescs2.shape))
    #print("orbkeypoints1: {}, orbdescriptors1: {}".format(len(orbkps1), orbdescs1.shape))
    #print("orbkeypoints2: {}, orbdescriptors2: {}".format(len(orbkps2), orbdescs2.shape))

    # Match the fezatures
    bfakaze = cv2.BFMatcher(cv2.NORM_HAMMING)
    bf = cv2.BFMatcher(cv2.NORM_L2)
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

    im3akaze = cv2.drawMatchesKnn(img_perfect, akazekps1, img_akaze, akazekps2, goodakaze[:100], None, flags=2)
    # cv2.imshow("AKAZE matching", im3akaze)
    goodakaze = np.asarray(goodakaze)
    # print("akaze")
    similarity_akaze = (goodakaze.shape[0]/feature_list[0])*100
    # print(similarity_akaze)

    # Apply ratio test on SIFT matches
    goodsift = []
    for m, n in siftmatches:
        if m.distance < 0.9 * n.distance:
            goodsift.append([m])

    im3sift = cv2.drawMatchesKnn(img_perfect, siftkps1, img_akaze, siftkps2, goodsift[:], None, flags=2)
    # cv2.imshow("SIFT matching", im3sift)
    goodsift = np.asarray(goodsift)
    # print("sift")
    similarity_sift = (goodsift.shape[0] / feature_list[1]) * 100
    # print(similarity_sift)


    # Apply ratio test on SURF matches
    goodsurf = []
    for m, n in surfmatches:
        if m.distance < 0.9 * n.distance:
            goodsurf.append([m])

    im3surf = cv2.drawMatchesKnn(img_perfect, surfkps1, img_akaze, surfkps2, goodsurf[:], None, flags=2)
    # cv2.imshow("SURF matching", im3surf)
    goodsurf = np.asarray(goodsurf)
    # print("surf")
    similarity_surf = (goodsurf.shape[0] / feature_list[2]) * 100
    # print(similarity_surf)

    # Apply ratio test on ORB matches
    goodorb = []
    for m, n in orbmatches:
        if m.distance < 0.9 * n.distance:
            goodorb.append([m])
    im3orb = cv2.drawMatchesKnn(img_perfect, orbkps1, img_akaze, orbkps2, goodorb[:], None, flags=2)
    # cv2.imshow("ORB matching", im3orb)
    goodorb = np.asarray(goodorb)
    # print("orb")
    similarity_orb = (goodorb.shape[0] / feature_list[3]) * 100
    # print(similarity_orb)

    # Apply ratio test on BRISK matches
    goodbrisk = []
    for m, n in briskmatches:
        if m.distance < 0.9 * n.distance:
            goodbrisk.append([m])

    im3brisk = cv2.drawMatchesKnn(img_perfect, briskkps1, img_akaze, briskkps2, goodbrisk[:], None, flags=2)
    # cv2.imshow("BRISK matching", im3brisk)
    goodbrisk = np.asarray(goodbrisk)
    # print("brisk")
    similarity_brisk = (goodbrisk.shape[0] / feature_list[4]) * 100
    # print(similarity_brisk)
    features_result = (similarity_akaze+similarity_brisk+similarity_orb+similarity_sift+similarity_surf)/5
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
    # cv2.imshow("image_akaze", img_akaze)
    # cv2.imshow("img_circle", img_circle)
    # cv2.waitKey(0)



# eval()