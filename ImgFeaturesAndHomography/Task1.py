"UBIT ='SAGARPOK' import numpy as np;np.random.seed(sum[ord(c) for c in SAGARPOK])"
import cv2
import os
import numpy as np
import random
UBIT = 'SAGARPOK'


print(np.random.uniform(0,1))
from matplotlib import pyplot as plt
MIN_MATCH_CNT =10

img=cv2.imread('mountain1.jpg')
gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img2=cv2.imread('mountain2.jpg')
gray2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

#Find keypoints This will solve 1.1
#keypoints in Left Image
img_sift = cv2.xfeatures2d.SIFT_create()
kp = img_sift.detect(gray,None)
#img=cv2.drawKeypoints(gray, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#keypoints in Right Image
img_sift2 = cv2.xfeatures2d.SIFT_create()
kp2 = img_sift2.detect(gray2,None)
#img2=cv2.drawKeypoints(gray2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

sift1 = cv2.drawKeypoints(img,kp,outImage=np.array([]), color=(255, 0, 0))
sift2 = cv2.drawKeypoints(img2,kp2,outImage=np.array([]), color=(255, 0, 0))

cv2.imwrite('task1_sift1.jpg', sift1)
cv2.imwrite('task1_sift2.jpg', sift2)



# find the keypoints and descriptors with SIFT,this will solve 1.2
kp1, desc1 = img_sift.detectAndCompute(img,None)
kp2, desc2 = img_sift2.detectAndCompute(img2,None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(desc1,desc2, k=2)
print(matches)
print('matches')

good = []
#print(matches)
good_without_list = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])  #[m] changes to m
        good_without_list.append(m)

#cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatches(img,kp1,img2,kp2,good_without_list,None,flags=2)

cv2.imwrite('task1_matches_knn.jpg', img3)


#Homography matrix from 1st image to 2ns image
if len(good_without_list)>MIN_MATCH_CNT:
    left_img_pts= np.float32([kp1[m.queryIdx].pt for m in good_without_list ]).reshape(-1,1,2)
    right_img_pts = np.float32([kp2[m.trainIdx].pt for m in good_without_list ]).reshape(-1,1,2)

    M, msk=cv2.findHomography(left_img_pts,right_img_pts,cv2.RANSAC,5.0)
    print(M)
    matchesMask =msk.ravel().tolist()
    Inlier_matches=[]
    i=0
    for m in good_without_list:
        if matchesMask[i]==1:
            Inlier_matches.append(m)
        i=i+1
    #Draw parameters
    np.random.seed(sum([ord(c) for c in UBIT]))
    Inlier_matches = np.random.choice(Inlier_matches, 10)
    draw_params = dict(matchesMask=None, singlePointColor=None, matchColor=(0, 0, 255), flags=2)
    Random10Matches = cv2.drawMatches(img, kp1, img2, kp2, Inlier_matches[:10], None, **draw_params)
    cv2.imwrite('task1_matches.jpg', Random10Matches)



else:
    print("Matches not found")
    matchesMask=None

#1.5 In progress...
WrappedImage1 = cv2.warpPerspective(img, M, (img.shape[1],img.shape[0]))
WrappedImage1[0:img.shape[0], 0:img.shape[1]] = img
images = []
WrappedImage2 = cv2.warpPerspective(img2, M, (img2.shape[1],img2.shape[0]))
WrappedImage2[0:img2.shape[0], 0:img2.shape[1]] = img2

images.append(WrappedImage1)
images.append(WrappedImage2)
try_use_gpu = False
stitcher = cv2.createStitcher(try_use_gpu)
status, pano = stitcher.stitch(images)
cv2.imwrite('task1_pano.jpg',pano)
