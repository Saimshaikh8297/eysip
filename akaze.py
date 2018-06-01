import cv2
import numpy as np
def kaze_match():
    # load the image and convert it to grayscale
    im1 = cv2.imread("plotlkt.jpg")
    im2 = cv2.imread("plotlkt.jpg")
    gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # initialize the AKAZE descriptor, then detect keypoints and extract
    # local invariant descriptors from the image
    sift = cv2.xfeatures2d.SIFT_create()
    surf = cv2.xfeatures2d.SURF_create()
    akaze = cv2.AKAZE_create()
    (akazekps1, akazedescs1) = akaze.detectAndCompute(gray1, None)
    (akazekps2, akazedescs2) = akaze.detectAndCompute(gray2, None)
    (siftkps1, siftdescs1) = sift.detectAndCompute(gray1, None)
    (siftkps2, siftdescs2) = sift.detectAndCompute(gray2, None)
    (surfkps1, surfdescs1) = surf.detectAndCompute(gray1, None)
    (surfkps2, surfdescs2) = surf.detectAndCompute(gray2, None)

    print("No of KeyPoints:")
    print("akazekeypoints1: {}, akazedescriptors1: {}".format(len(akazekps1), akazedescs1.shape))
    print("akazekeypoints2: {}, akazedescriptors2: {}".format(len(akazekps2), akazedescs2.shape))
    print("siftkeypoints1: {}, siftdescriptors1: {}".format(len(siftkps1), siftdescs1.shape))
    print("siftkeypoints2: {}, siftdescriptors2: {}".format(len(siftkps2), siftdescs2.shape))
    print("surfkeypoints1: {}, surfdescriptors1: {}".format(len(surfkps1), surfdescs1.shape))
    print("surfkeypoints2: {}, surfdescriptors2: {}".format(len(surfkps2), surfdescs2.shape))

    # Match the features
    bfakaze = cv2.BFMatcher(cv2.NORM_HAMMING)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    akazematches = bfakaze.knnMatch(akazedescs1,akazedescs2, k=2)
    siftmatches = bf.knnMatch(siftdescs1, siftdescs2, k=2)
    surfmatches = bf.knnMatch(surfdescs1, surfdescs2, k=2)

    # Apply ratio test on AKAZE matches
    goodakaze = []
    for m,n in akazematches:
        if m.distance < 0.9*n.distance:
            goodakaze.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    im3akaze = cv2.drawMatchesKnn(im1, akazekps1, im2, akazekps2, goodakaze[1:20], None, flags=2)
    cv2.imshow("AKAZE matching", im3akaze)
    goodakaze=np.asarray(goodakaze)
    print(goodakaze.shape)

    # Apply ratio test on SIFT matches
    goodsift = []
    for m, n in siftmatches:
        if m.distance < 0.9 * n.distance:
            goodsift.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    im3sift = cv2.drawMatchesKnn(im1, siftkps1, im2, siftkps2, goodsift[1:20], None, flags=2)
    cv2.imshow("SIFT matching", im3sift)
    goodsift = np.asarray(goodsift)
    print(goodsift.shape)

    # Apply ratio test on SURF matches
    goodsurf = []
    for m, n in surfmatches:
        if m.distance < 0.9 * n.distance:
            goodsurf.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    im3surf = cv2.drawMatchesKnn(im1, surfkps1, im2, surfkps2, goodsurf[1:20], None, flags=2)
    cv2.imshow("SURF matching", im3surf)
    goodsurf = np.asarray(goodsurf)
    print(goodsurf.shape)

    cv2.waitKey(0)

kaze_match()