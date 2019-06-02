import cv2
import numpy as np

from matplotlib import pyplot as plt
import random
UBIT = 'SAGARPOK'
MIN_MATCH_CNT =10

def draw_lines(img1, img2, lines, pts1, pts2):
    r = img1.shape[0]
    c = img1.shape[1]
    #img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    #img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)

    '''tuple1=(255,0,0),
    tuple2=(0,255,0)
    tuple3 = (0,0,255)
    tuple4 = (255, 255, 0)
    tuple5 = (255, 0, 255)
    tuple6=(0,255,255)
    tuple7 = (255,255,255)
    tuple8 = (255, 0, 200)
    tuple9 = (255, 100, 255)
    tuple10 = (255, 255, 100)
    tuple11=(tuple1,tuple2,tuple3,tuple4,tuple5,tuple6,tuple7,tuple8,tuple9,tuple10)

    color=list(tuple11)'''
    color=[[255,0,0],[0,255,0],[0,0,255],[255,0,0],[255, 255, 0],[255, 0, 255],[0, 255, 255],[255, 255, 255],[255, 0, 200],[255, 0, 100],[255, 255, 100]]
    i=0
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color[i],1)
        #img2 = cv2.circle(img2,tuple(pt2),5,color[i],-1)
        i=i+1
    return img1,img2

img_left=cv2.imread('tsucuba_left.png')
gray= cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)

img_right=cv2.imread('tsucuba_right.png')
gray2=cv2.cvtColor(img_right,cv2.COLOR_BGR2GRAY)

#Find keypoints This will solve 2.1
#keypoints in Left Image
img_sift = cv2.xfeatures2d.SIFT_create()
kp = img_sift.detect(gray,None)
img=cv2.drawKeypoints(gray, kp, None, flags=2)
cv2.imwrite('task2_sift1.jpg', img)

#keypoints in Right Image
img_sift2 = cv2.xfeatures2d.SIFT_create()
kp2 = img_sift2.detect(img_right,None)
img2=cv2.drawKeypoints(gray2, kp2, None, flags=2)
cv2.imwrite('task2_sift2.jpg', img2)

#find the keypoints and descriptors with SIFT,this will solve 1.2
kp1, des1 = img_sift.detectAndCompute(gray,None)
kp2, des2 = img_sift.detectAndCompute(gray2,None)

#BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
good = []
good_without_list = []
pts1=[]
pts2=[]
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])  #[m] changes to m
        good_without_list.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

img3 = cv2.drawMatchesKnn(img,kp1,img2,kp2,good,None,flags=2)
#plt.imshow(img3),plt.show()
cv2.imwrite('task2_matches_knn.jpg', img3)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
Fundamental_mat, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC)
print('Fundamental matrix')
print(Fundamental_mat)
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]
matchesMask =mask.ravel().tolist()
# cv2.drawMatchesKnn expects list of  aslists matches.
Inlier_matches=[]
i=0
np.random.seed(sum([ord(c) for c in UBIT]))
#InlierMatches = np.random.choice(Inlier_matches,10)
for m in good_without_list:
    if matchesMask[i]==1:
        Inlier_matches.append(m)
    i=i+1
    #Draw parameters
np.random.seed(sum([ord(c) for c in UBIT]))
Inlier_matches = np.random.choice(Inlier_matches,10)

points1 = []
points2 = []
for m in Inlier_matches:
    points2.append(kp2[m.trainIdx].pt)
    points1.append(kp1[m.queryIdx].pt)

points1 = points1[:10]
points2 = points2[:10]
points1 = np.int32(points1)
points2 = np.int32(points2)

lines2 = cv2.computeCorrespondEpilines(points1.reshape(-1,1,2), 1,Fundamental_mat)
lines2 = lines2.reshape(-1,3)

img3,img4 = draw_lines(img_left,img_right,lines2,points1,points2)
cv2.imwrite('task2_epi_right.jpg',img3)

lines3 = cv2.computeCorrespondEpilines(points2.reshape(-1,1,2), 1,Fundamental_mat)
lines3 = lines3.reshape(-1,3)
img5,img6 = draw_lines(img_right,img_left,lines3,points2,points1)

cv2.imwrite('task2_epi_left.jpg',img5)
#Disparity Map using StereoSGBM_Create

window_size = 3
min_disp = 16
num_disp = 112-min_disp
stereo = cv2.StereoSGBM_create(minDisparity = min_disp,numDisparities = num_disp,blockSize = 16,P1 = 8*3*window_size**2,P2 = 32*3*window_size**2,disp12MaxDiff = 1,uniquenessRatio = 10,
          speckleWindowSize = 100,speckleRange = 32)

disp = stereo.compute(gray, gray2).astype(np.float32) / 16.0

mask = disp > disp.min()
cv2.imshow('left', gray)
cv2.imshow('disparity', (disp-min_disp)/num_disp)
cv2.imwrite('task2_disparity.jpg',255*(disp-min_disp)/num_disp)
cv2.waitKey()
cv2.destroyAllWindows()
