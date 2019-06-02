import cv2
import numpy as np
import math

#Read Input Image
img_col=cv2.imread("G:\Task2.jpg")
image=cv2.imread("G:\Task2.jpg",0)
cv2.imshow('test',image)

r = image.shape[0]  # Row
c = image.shape[1]  #column

#Resize gray scale images
resize_1=image[0:len(image):2,0:len(image[0]):2]
resize_2=image[0:len(image):4,0:len(image[0]):4]
resize_3=image[0:len(image):8,0:len(image[0]):8]

#Resize color images
resize_1_color=img_col[0:len(image):2,0:len(image[0]):2]
resize_2_color=img_col[0:len(image):4,0:len(image[0]):4]
resize_3_color=img_col[0:len(image):8,0:len(image[0]):8]
r1_1=resize_1.shape[0]
c1_1=resize_1.shape[1]
print('r1_1',r1_1)
print('c1_1',c1_1)

r2_1=resize_2_color.shape[0]
r2_2=resize_2_color.shape[1]
print('r2_1',r2_1)
print('r2_2',r2_2)

cv2.imwrite('G:\CVIP Project//resize_1_color.png', resize_1_color)
cv2.imwrite('G:\CVIP Project//resize_2_color.png', resize_2_color)
#Gaussian Kernel function
def Gaussian(sigma):
    m=3
    n=3
    Gaussian_blur = np.zeros((7,7))

    for i in range(-m, m):
        for j in range(-n, n):
            x1=(sigma**2)*(2*3.14)
            p=float(i**2+j**2)
           # print("P \n",p)
            q=2*sigma**2
            x2=math.e**(p/q)
            Gaussian_blur[i,j]=x1*(1/x2)

    return Gaussian_blur


sumx=0.0
n=np.zeros(image.shape)

#Convolution function

def convolution(g,x):
    n = np.zeros(x.shape)
    for i in range(3, x.shape[0]-3):
        for j in range(3, x.shape[1]-3):

            sumx = (g[0][0] * x[i-3][j-3]) + ((g[0][1]) * x[i-3][j-2]) + ((g[0][2]) * x[i-3][j-1])+((g[0][3]) * x[i-3][j])\
                    +((g[0][4]) * x[i-3][j+1])+((g[0][5]) * x[i-3][j+2])+((g[0][6]) * x[i-3][j+3])\
                     +(g[1][0] * x[i - 2][j - 3]) + ((g[0][1]) * x[i - 2][j - 2]) + ((g[0][2]) * x[i - 2][j - 1]) + \
                        (g[0][3]) * x[i - 2][j]
            +((g[0][4]) * x[i - 2][j + 1]) + ((g[0][5]) * x[i - 2][j + 2]) + ((g[0][6]) * x[i - 2][j + 2]) \

            +(g[2][0] * x[i - 1][j - 3]) + ((g[2][1]) * x[i - 1][j - 2]) + ((g[2][2]) * x[i - 1][j - 1]) + \
                        (g[2][3]) * x[i - 1][j]
            +((g[2][4]) * x[i - 1][j + 1]) + ((g[2][5]) * x[i - 1][j + 2]) + ((g[2][6]) * x[i - 1][j + 2])

            (g[3][0] * x[i][j - 3]) + ((g[3][1]) * x[i][j - 2]) + ((g[3][2]) * x[i][j - 1]) + ((g[3][3]) * x[i][j])\
            +((g[3][4]) * x[i][j + 1]) + ((g[3][5]) * x[i][j + 2]) + ((g[3][6]) * x[i][j + 2]) \

            +(g[4][0] * x[i + 1][j - 3]) + ((g[4][1]) * x[i + 1][j - 2]) + ((g[4][2]) * x[i + 1][j - 1]) \
                    +(g[4][3]) * x[i +1][j]
            +((g[4][4]) * x[i + 1][j + 1]) + ((g[4][5]) * x[i + 1][j + 2]) + ((g[4][6]) * x[i + 1][j + 2]) \

            +(g[5][0] * x[i + 2][j - 3]) + ((g[5][1]) * x[i + 2][j - 2]) + ((g[5][2]) * x[i + 2][j - 1]) \
                    +(g[5][3]) * x[i + 2][j]
            +((g[5][4]) * x[i + 2][j + 1]) + ((g[5][5]) * x[i + 2][j + 2]) + ((g[5][6]) * x[i + 2][j + 2]) \


            +(g[6][0] * x[i + 3][j - 3]) + ((g[6][1]) * x[i + 3][j - 2]) + ((g[6][2]) * x[i + 3][j - 1]) \
                    +(g[6][3]) * x[i + 3][j]
            +((g[6][4]) * x[i + 3][j + 1]) + ((g[6][5]) * x[i + 3][j + 2]) + ((g[6][6]) * x[i + 3][j + 2])
            n[i][j] = sumx
    return n

#octave1 1st image
g1_1=Gaussian(1/math.sqrt(2))
g1_1_convolve=(g1_1 - np.min(g1_1)) / (np.max(g1_1) - np.min(g1_1))
conv1_1=convolution(g1_1_convolve,image)
# Eliminate zero values with method 1
blur1_1 = (conv1_1 - np.min(conv1_1)) / (np.max(conv1_1) - np.min(conv1_1))
#cv2.namedWindow('Blurred Image', cv2.WINDOW_NORMAL)
cv2.imshow('octave1 blur1', blur1_1)
status = cv2.imwrite('G:\CVIP Project//blur1_1.png', 255*blur1_1)
print("Image written to file-system : ", status)


#octave1 2nd image
g1_2=Gaussian(1)
g1_2_convolve=(g1_2 - np.min(g1_1)) / (np.max(g1_2) - np.min(g1_2))
conv1_2=convolution(g1_2_convolve,image)
# Eliminate zero values with method 1
blur1_2 = (conv1_2 - np.min(conv1_2)) / (np.max(conv1_2) - np.min(conv1_2))
#cv2.namedWindow('Blurred Image', cv2.WINDOW_NORMAL)
cv2.imshow('octave1 blur2', blur1_2)
status = cv2.imwrite('G:\CVIP Project//blur1_2.png', 255*blur1_2)
print("Image written to file-system : ", status)

#octave1 3nd image
g1_3=Gaussian(math.sqrt(2))
g1_3_convolve=(g1_2 - np.min(g1_3)) / (np.max(g1_3) - np.min(g1_3))
conv1_3=convolution(g1_3_convolve,image)
# Eliminate zero values with method 1
blur1_3 = (conv1_3 - np.min(conv1_3)) / (np.max(conv1_3) - np.min(conv1_3))
#cv2.namedWindow('Blurred Image', cv2.WINDOW_NORMAL)
cv2.imshow('octave1 blur3', blur1_3)
status = cv2.imwrite('G:\CVIP Project//blur1_3.png', 255*blur1_3)
print("Image written to file-system : ", status)

#octave1 4th image
g1_4=Gaussian(2)
g1_4_convolve=(g1_4 - np.min(g1_4)) / (np.max(g1_4) - np.min(g1_4))
conv1_4=convolution(g1_4_convolve,image)
# Eliminate zero values with method 1
blur1_4 = (conv1_4 - np.min(conv1_4)) / (np.max(conv1_4) - np.min(conv1_4))
#cv2.namedWindow('Blurred Image', cv2.WINDOW_NORMAL)
cv2.imshow('octave1 blur4', blur1_4)
status = cv2.imwrite('G:\CVIP Project//blur1_4.png', 255*blur1_4)
print("Image written to file-system : ", status)

#octave1 5th image
g1_5=Gaussian(2*math.sqrt(2))
g1_5_convolve=(g1_5 - np.min(g1_5)) / (np.max(g1_5) - np.min(g1_5))
conv1_5=convolution(g1_5_convolve,image)
# Eliminate zero values with method 1
blur1_5 = (conv1_5 - np.min(conv1_5)) / (np.max(conv1_5) - np.min(conv1_5))
#cv2.namedWindow('Blurred Image 5', cv2.WINDOW_NORMAL)
cv2.imshow('octave1 blur5', blur1_5)
status = cv2.imwrite('G:\CVIP Project//blur1_5.png', 255*blur1_5)
print("Image written to file-system : ", status)

#octave1 DOG1
OCT1_DOG_Change_1=blur1_2-blur1_1
# Eliminate zero values with method 1
OCT1_DOG1 = (OCT1_DOG_Change_1 - np.min(OCT1_DOG_Change_1)) / (np.max(OCT1_DOG_Change_1) - np.min(OCT1_DOG_Change_1))
#cv2.namedWindow('DOG Image 1', cv2.WINDOW_NORMAL)
cv2.imshow('octave1 DOG 1', OCT1_DOG1)
status = cv2.imwrite('G:\CVIP Project//OCT1_DOG1.png', 255*OCT1_DOG1)
print("Image written to file-system : ", status)

#octave1 DOG2
OCT1_DOG_Change_2=blur1_3-blur1_2
# Eliminate zero values with method 1
OCT1_DOG2 = (OCT1_DOG_Change_2 - np.min(OCT1_DOG_Change_2)) / (np.max(OCT1_DOG_Change_2) - np.min(OCT1_DOG_Change_2))
#cv2.namedWindow('DOG Image 2', cv2.WINDOW_NORMAL)
cv2.imshow('octave1 DOG 2', OCT1_DOG2)
status = cv2.imwrite('G:\CVIP Project//OCT1_DOG2.png', 255*OCT1_DOG2)
print("Image written to file-system : ", status)

#octave1 DOG3
OCT1_DOG_Change_3=blur1_4-blur1_3
# Eliminate zero values with method 1
OCT1_DOG3 = (OCT1_DOG_Change_3 - np.min(OCT1_DOG_Change_3)) / (np.max(OCT1_DOG_Change_3) - np.min(OCT1_DOG_Change_3))
#cv2.namedWindow('DOG Image 3', cv2.WINDOW_NORMAL)
cv2.imshow('octave1 DOG 3', OCT1_DOG3)
status = cv2.imwrite('G:\CVIP Project//OCT1_DOG3.png', 255*OCT1_DOG3)
print("Image written to file-system : ", status)

#octave1 DOG4
OCT1_DOG_Change_4=blur1_5-blur1_4
# Eliminate zero values with method 1
OCT1_DOG4 = (OCT1_DOG_Change_4 - np.min(OCT1_DOG_Change_4)) / (np.max(OCT1_DOG_Change_4) - np.min(OCT1_DOG_Change_4))
#cv2.imwrite("OCT1_DOG4 Write",OCT1_DOG4)
#cv2.namedWindow('DOG Image 4', cv2.WINDOW_NORMAL)
cv2.imshow('octave1 DOG 4', OCT1_DOG4)
status = cv2.imwrite('G:\CVIP Project//OCT1_DOG4.png', 255*OCT1_DOG4)
print("Image written to file-system : ", status)


def keypoint1(OCT1,OCT2,OCT3):
        octave1_key1 = np.zeros(OCT2.shape);
        for i in range(1,OCT1.shape[0]-1):
            for j in range(1,OCT1.shape[1]-1):
                    if(OCT2[i][j]>OCT2[i-1][j-1] and OCT2[i][j]>OCT2[i-1][j+1] and OCT2[i][j]>OCT2[i][j-1]
                        and OCT2[i][j] > OCT2[i][j+1] and OCT2[i][j]>OCT2[i+1][j-1]
                        and OCT2[i][j]>OCT2[i+1][j] and OCT2[i][j] > OCT2[i+1][j+1]
                        and OCT2[i][j]>OCT1[i-1][j-1] and OCT2[i][j]>OCT1[i-1][j] and OCT2[i][j]>OCT1[i-1][j+1]
                        and OCT2[i][j]>OCT1[i+1][j-1] and OCT2[i][j]>OCT1[i+1][j] and OCT2[i][j]>OCT1[i+1][j+1]
                        and OCT2[i][j] >OCT1[i+1][j-1] and OCT2[i][j]>OCT1[i+1][j] and OCT2[i][j]>OCT1[i+1][j+1]
                        and OCT2[i][j]>OCT3[i-1][j-1] and OCT2[i][j]>OCT3[i-1][j] and OCT2[i][j]>OCT3[i-1][j+1]
                        and OCT2[i][j]>OCT3[i+1][j-1] and OCT2[i][j]>OCT3[i+1][j] and OCT2[i][j]>OCT3[i+1][j+1]
                        and OCT2[i][j] > OCT3[i+1][j-1] and OCT2[i][j]>OCT3[i+1][j] and OCT2[i][j]>OCT3[i+1][j+1]
                        ):
                        octave1_key1[i][j] = OCT2[i][j]
        return octave1_key1


def keypoint2(OCT2, OCT3, OCT4):
    octave1_key2 = np.zeros(OCT3.shape);
    for i in range(1, OCT3.shape[0] - 1):
        for j in range(1, OCT3.shape[1] - 1):
            if (OCT3[i][j] < OCT3[i - 1][j - 1] and OCT3[i][j] < OCT3[i - 1][j] and OCT3[i][j] < OCT3[i][j + 1]
                    and OCT3[i][j] < OCT3[i][j - 1] and OCT3[i][j] < OCT3[i + 1][j + 1]
                    and OCT3[i][j] < OCT3[i + 1][j - 1] and OCT3[i][j] < OCT3[i + 1][j] and
                    OCT3[i][j] < OCT3[i + 1][j + 1]
                    and OCT3[i][j] < OCT2[i - 1][j - 1] and OCT2[i][j] < OCT2[i - 1][j] and
                    OCT3[i][j] < OCT2[i - 1][j + 1]
                    and OCT3[i][j] < OCT2[i + 1][j - 1] and OCT3[i][j] < OCT2[i + 1][j] and
                    OCT3[i][j] < OCT2[i + 1][j + 1]
                    and OCT3[i][j] < OCT2[i + 1][j - 1] and OCT3[i][j] < OCT2[i + 1][j] and
                    OCT3[i][j] < OCT2[i + 1][j + 1]
                    and OCT3[i][j] < OCT4[i - 1][j - 1] and OCT3[i][j] < OCT4[i - 1][j] and
                    OCT3[i][j] < OCT4[i - 1][j + 1]
                    and OCT3[i][j] < OCT4[i + 1][j - 1] and OCT3[i][j] < OCT4[i + 1][j] and
                    OCT3[i][j] < OCT4[i + 1][j + 1]
                    and OCT3[i][j] < OCT4[i + 1][j - 1] and OCT3[i][j] < OCT4[i + 1][j] and
                    OCT3[i][j] < OCT4[i + 1][j + 1]
            ):
                octave1_key2[i][j] = OCT3[i][j]
    return octave1_key2


#Octave1 key point detection

octave1_key1=np.zeros(OCT1_DOG3.shape);
octave1_key2=np.zeros(OCT1_DOG3.shape);
octave1_key1=keypoint1(OCT1_DOG1,OCT1_DOG2,OCT1_DOG3)
octave1_key2=keypoint2(OCT1_DOG2,OCT1_DOG3,OCT1_DOG4)
octave_key01 = (octave1_key1 - np.min(octave1_key1)) / (np.max(octave1_key1) - np.min(octave1_key1))
octave_key02 = (octave1_key2 - np.min(octave1_key2)) / (np.max(octave1_key2) - np.min(octave1_key2))

cv2.imshow("octave1 Image Key detection_1",octave_key01)
cv2.imshow("octave1 Image Key detection_2",octave_key02)
#cv2.imshow('octave key point',octave1_key)
status = cv2.imwrite('G:\CVIP Project//octave_key01.png', 255*octave_key01)
print("Image written to file-system : ", status)
status = cv2.imwrite('G:\CVIP Project//octave_key02.png', 255*octave_key02)
print("Image written to file-system : ", status)
#Key detection in original image
for i in range(1,octave_key01.shape[0]):
    for j in range(1,octave_key01.shape[1]):
        if(octave_key01[i][j]>0.5):
            img_col[i][j]=255;

cv2.imshow("octave1 merge Image Key detection_1",img_col)
status = cv2.imwrite('G:\CVIP Project//octave_key01_img_col.png', img_col)
print("Image written to file-system : ", status)
for i in range(1,octave_key01.shape[0]):
    for j in range(1,octave_key01.shape[1]):
        if(octave_key01[i][j]>0.5):
            img_col[i][j]=255;

cv2.imshow("octave1 merge Image Key detection_2",img_col)
status = cv2.imwrite('G:\CVIP Project//octave_key02_img_col.png', img_col)

#octave2 1st image
g2_1=Gaussian(math.sqrt(2))
g2_1_convolve=(g2_1 - np.min(g2_1)) / (np.max(g2_1) - np.min(g2_1))
conv2_1=convolution(g2_1_convolve,resize_1)

# Eliminate zero values with method 1
blur2_1 = (conv2_1 - np.min(conv2_1)) / (np.max(conv2_1) - np.min(conv2_1))
#cv2.namedWindow('Blurred Image octave 2_1', cv2.WINDOW_NORMAL)
cv2.imshow('octave2 blur2_1', blur2_1)
status = cv2.imwrite('G:\CVIP Project//blur2_1.png', 255*blur2_1)
#octave2 2nd image
g2_2=Gaussian(2)
conv2_2=convolution(g2_2,resize_1)
# Eliminate zero values with method 1
blur2_2 = (conv2_2 - np.min(conv2_2)) / (np.max(conv2_2) - np.min(conv2_2))
#cv2.namedWindow('Blurred Image octave 2_2', cv2.WINDOW_NORMAL)
cv2.imshow('octave2 blur2_2', blur2_2)
status = cv2.imwrite('G:\CVIP Project//blur2_2.png', 255*blur2_2)

#octave2 3rd image
g2_3=Gaussian(2*math.sqrt(2))
conv2_3=convolution(g2_3,resize_1)
# Eliminate zero values with method 1
blur2_3 = (conv2_3 - np.min(conv2_3)) / (np.max(conv2_3) - np.min(conv2_3))
#cv2.namedWindow('Blurred Image octave 2_3', cv2.WINDOW_NORMAL)
cv2.imshow('octave2 blur2_3', blur2_3)
status = cv2.imwrite('G:\CVIP Project//blur2_3.png', 255*blur2_3)
#octave2 4th image
g2_4=Gaussian(4)
conv2_4=convolution(g2_4,resize_1)
# Eliminate zero values with method 1
blur2_4 = (conv2_4 - np.min(conv2_4)) / (np.max(conv2_4) - np.min(conv2_4))
#cv2.namedWindow('Blurred Image octave 2_4', cv2.WINDOW_NORMAL)
cv2.imshow('octave2 blur2_4', blur2_4)
status = cv2.imwrite('G:\CVIP Project//blur2_4.png', 255*blur2_4)

#octave2 5th image
g2_5=Gaussian(4*math.sqrt(2))
conv2_5=convolution(g2_5,resize_1)
# Eliminate zero values with method 1
blur2_5 = (conv2_5 - np.min(conv2_5)) / (np.max(conv2_5) - np.min(conv2_5))
#cv2.namedWindow('Blurred Image octave 2_5', cv2.WINDOW_NORMAL)
cv2.imshow('octave2 blur2_5', blur2_5)
status = cv2.imwrite('G:\CVIP Project//blur2_5.png', 255*blur2_5)

#octave2 DOG2
OCT2_DOG_Change_1=blur2_2-blur2_1
# Eliminate zero values with method 1
OCT2_DOG1 = (OCT2_DOG_Change_1 - np.min(OCT2_DOG_Change_1)) / (np.max(OCT2_DOG_Change_1) - np.min(OCT2_DOG_Change_1))
#cv2.namedWindow('DOG Image 1', cv2.WINDOW_NORMAL)
cv2.imshow('octave 2 DOG 1', OCT2_DOG1)
status = cv2.imwrite('G:\CVIP Project//OCT2_DOG1.png', 255*OCT2_DOG1)

#octave1 DOG2
OCT2_DOG_Change_2=blur2_3-blur2_2
# Eliminate zero values with method 1
OCT2_DOG2 = (OCT2_DOG_Change_2 - np.min(OCT2_DOG_Change_2)) / (np.max(OCT2_DOG_Change_2) - np.min(OCT2_DOG_Change_2))
#cv2.namedWindow('DOG Image 2', cv2.WINDOW_NORMAL)
cv2.imshow(' octave 2 DOG 2', OCT2_DOG2)
status = cv2.imwrite('G:\CVIP Project//OCT2_DOG2.png', 255*OCT2_DOG2)

#octave2 DOG3
OCT2_DOG_Change_3=blur2_4-blur2_3
# Eliminate zero values with method 1
OCT2_DOG3 = (OCT2_DOG_Change_3 - np.min(OCT2_DOG_Change_3)) / (np.max(OCT2_DOG_Change_3) - np.min(OCT2_DOG_Change_3))
#cv2.namedWindow('DOG Image 3', cv2.WINDOW_NORMAL)
cv2.imshow(' octave 2 DOG 3', OCT2_DOG3)
status = cv2.imwrite('G:\CVIP Project//OCT2_DOG3.png', 255*OCT2_DOG3)
#octave2 DOG4
OCT2_DOG_Change_4=blur2_5-blur2_4
# Eliminate zero values with method 1
OCT2_DOG4 = (OCT2_DOG_Change_4 - np.min(OCT2_DOG_Change_4)) / (np.max(OCT2_DOG_Change_4) - np.min(OCT2_DOG_Change_4))

#cv2.namedWindow('DOG Image 4', cv2.WINDOW_NORMAL)
cv2.imshow(' octave 2 DOG 4', OCT2_DOG4)
#cv2.imwrite("OCT1_DOG4 Write",OCT1_DOG4)
status = cv2.imwrite('G:\CVIP Project//OCT2_DOG4.png', 255*OCT2_DOG4)

#octave2 key detection
octave2_key1=np.zeros(OCT2_DOG3.shape);
octave2_key2=np.zeros(OCT2_DOG3.shape);
octave2_key1=keypoint1(OCT2_DOG1,OCT2_DOG2,OCT2_DOG3)
octave2_key2=keypoint2(OCT2_DOG2,OCT2_DOG3,OCT2_DOG4)
octave2_key01 = (octave2_key1 - np.min(octave2_key1)) / (np.max(octave2_key1) - np.min(octave2_key1))
octave2_key02 = (octave2_key2 - np.min(octave2_key2)) / (np.max(octave2_key2) - np.min(octave2_key2))

cv2.imshow("octave2 Image Key detection_1",octave2_key01)
cv2.imshow("octave2 Image Key detection_2",octave2_key02)
cv2.imwrite('G:\CVIP Project//octave2_key01.png', octave2_key01)
cv2.imwrite('G:\CVIP Project//octave2_key02.png', octave2_key02)

#Key detection in original image
print('key points oct2 in set 1')
for i in range(1,octave2_key01.shape[0]):
    for j in range(1,octave2_key01.shape[1]):
        if(octave2_key01[i][j]>0.5):
            resize_1_color[i][j]=255
            print("x1,y1",i,j);
cv2.imshow("octave2 merge Image Key detection_1",resize_1_color)
status = cv2.imwrite('G:\CVIP Project//octave2_key01_resize_1_color.png', resize_1_color)
print(np.asarray(octave2_key01))

print('key points oct 2 in set 2')
for i in range(1,octave2_key01.shape[0]):
    for j in range(1,octave2_key01.shape[1]):
        if(octave2_key01[i][j]>0.5):
            resize_1_color[i][j]=255;
            print("x2,y2", i, j);
cv2.imshow("octave2 merge Image Key detection_2",resize_1_color)
status = cv2.imwrite('G:\CVIP Project//octave2_key02_resize_1_color.png', resize_1_color)

#print('octave 2 key points 1 matrix')
#print(np.matrix(octave2_key01))

#print('octave 2 key points 2 matrix')
#print(np.matrix(octave2_key02))

#octave 3 Blur 1 image
g3_1=Gaussian(2*math.sqrt(2))
blur3_1=np.zeros(resize_2.shape)
conv3_1=np.zeros(resize_2.shape)
conv3_1=convolution(g3_1,resize_2)
# Eliminate zero values with method 1

blur3_1 = (conv3_1 - np.min(conv3_1)) / (np.max(conv3_1) - np.min(conv3_1))
#cv2.namedWindow('Blurred Image octave 3_1', cv2.WINDOW_NORMAL)
cv2.imshow('octave3 blur3_1', blur3_1)
status = cv2.imwrite('G:\CVIP Project//blur3_1.png', 255*blur3_1)

#octave 3 Blur 2 image
g3_2=Gaussian(4)
conv3_2=convolution(g3_2,resize_2)
blur3_2=np.zeros(resize_2.shape)
# Eliminate zero values with method 1
blur3_2 = (conv3_2 - np.min(conv3_2)) / (np.max(conv3_2) - np.min(conv3_2))
#cv2.namedWindow('Blurred Image octave 3_1', cv2.WINDOW_NORMAL)
cv2.imshow('octave3 blur3_2', blur3_2)
status = cv2.imwrite('G:\CVIP Project//blur3_2.png', 255*blur3_2)

#octave 3 Blur 3 image
g3_3=Gaussian(4*math.sqrt(2))
conv3_3=convolution(g3_3,resize_2)
blur3_3=np.zeros(resize_2.shape)
# Eliminate zero values with method 1
blur3_3 = (conv3_3 - np.min(conv3_3)) / (np.max(conv3_3) - np.min(conv3_3))
#cv2.namedWindow('Blurred Image octave 3_1', cv2.WINDOW_NORMAL)
cv2.imshow('octave3 blur3_3', blur3_3)
status = cv2.imwrite('G:\CVIP Project//blur3_3.png', 255*blur3_3)

#octave 3 Blur 4 image
g3_4=Gaussian(8)
conv3_4=convolution(g3_4,resize_2)
blur3_4=np.zeros(resize_2.shape)
# Eliminate zero values with method 1
blur3_4 = (conv3_4 - np.min(conv3_4)) / (np.max(conv3_4) - np.min(conv3_4))
#cv2.namedWindow('Blurred Image octave 3_1', cv2.WINDOW_NORMAL)
cv2.imshow('octave3 blur3_4', blur3_4)
status = cv2.imwrite('G:\CVIP Project//blur3_4.png', 255*blur3_4)

#octave 3 Blur 5 image
g3_5=Gaussian(8*math.sqrt(2))
conv3_5=convolution(g3_5,resize_2)
blur3_5=np.zeros(resize_2.shape)
# Eliminate zero values with method 1
blur3_5 = (conv3_5 - np.min(conv3_5)) / (np.max(conv3_5) - np.min(conv3_5))
#cv2.namedWindow('Blurred Image octave 3_1', cv2.WINDOW_NORMAL)
cv2.imshow('octave3 blur3_5', blur3_5)
status = cv2.imwrite('G:\CVIP Project//blur3_5.png', 255*blur3_5)

#octave3 DOG1
OCT3_DOG_Change_1=blur3_2-blur3_1
# Eliminate zero values with method 1
OCT3_DOG1 = (OCT3_DOG_Change_1 - np.min(OCT3_DOG_Change_1)) / (np.max(OCT3_DOG_Change_1) - np.min(OCT3_DOG_Change_1))
#cv2.namedWindow('DOG Image 1', cv2.WINDOW_NORMAL)
cv2.imshow('octave 3 DOG 1', OCT3_DOG1)
status = cv2.imwrite('G:\CVIP Project//OCT3_DOG1.png', 255*OCT3_DOG1)

#octave3 DOG2
OCT3_DOG_Change_2=blur3_3-blur3_2
# Eliminate zero values with method 1
OCT3_DOG2 = (OCT3_DOG_Change_2 - np.min(OCT3_DOG_Change_2)) / (np.max(OCT3_DOG_Change_2) - np.min(OCT3_DOG_Change_2))
#cv2.namedWindow('DOG Image 2', cv2.WINDOW_NORMAL)
cv2.imshow(' octave 3 DOG 2', OCT3_DOG2)
status = cv2.imwrite('G:\CVIP Project//OCT3_DOG2.png', 255*OCT3_DOG2)
#octave3 DOG3

OCT3_DOG_Change_3=blur3_4-blur3_3
# Eliminate zero values with method 1
OCT3_DOG3 = (OCT3_DOG_Change_3 - np.min(OCT3_DOG_Change_3)) / (np.max(OCT3_DOG_Change_3) - np.min(OCT3_DOG_Change_3))
#cv2.namedWindow('DOG Image 3', cv2.WINDOW_NORMAL)
cv2.imshow(' octave 3 DOG 3', OCT3_DOG3)
status = cv2.imwrite('G:\CVIP Project//OCT3_DOG3.png', 255*OCT3_DOG3)

#octave3 DOG4
OCT3_DOG_Change_4=blur3_5-blur3_4
# Eliminate zero values with method 1
OCT3_DOG4 = (OCT3_DOG_Change_4 - np.min(OCT3_DOG_Change_4)) / (np.max(OCT3_DOG_Change_4) - np.min(OCT3_DOG_Change_4))
#cv2.imwrite("OCT1_DOG4 Write",OCT1_DOG4)
cv2.imshow(' octave 3 DOG 4', OCT3_DOG4)
status = cv2.imwrite('G:\CVIP Project//OCT3_DOG4.png', 255*OCT3_DOG4)

#octave3 key detection
octave3_key1=np.zeros(OCT3_DOG3.shape);
octave3_key2=np.zeros(OCT3_DOG3.shape);
octave3_key1=keypoint1(OCT3_DOG1,OCT3_DOG2,OCT3_DOG3)
octave3_key2=keypoint2(OCT3_DOG2,OCT3_DOG3,OCT3_DOG4)
octave3_key01 = (octave3_key1 - np.min(octave3_key1)) / (np.max(octave3_key1) - np.min(octave3_key1))
octave3_key02 = (octave3_key2 - np.min(octave3_key2)) / (np.max(octave3_key2) - np.min(octave3_key2))


#print("octave 3 key")
#print(255*octave3_key01)

cv2.imshow("octave3 Image Key detection_1",octave3_key01)
cv2.imshow("octave3 Image Key detection_2",octave3_key02)
cv2.imwrite('G:\CVIP Project//octave3_key01_Gray.png', octave3_key01)
cv2.imwrite('G:\CVIP Project//octave3_key02_Gray.png', octave3_key02)
print('dim')
print(octave3_key01.shape[0])
print(octave3_key01.shape[1])
r_3=octave3_key01.shape[0]
c_3=octave3_key01.shape[1]

print('key points oct 2 in set 1')
for i in range(1,r_3):
    for j in range(1,c_3):
        if(octave3_key01[i][j]>0.9):
            resize_2_color[i][j]=255;
            print('x1,y1',i,j)
cv2.imshow("octave3 merge Image Key detection_1",resize_2_color)
status = cv2.imwrite('G:\CVIP Project//octave3_key01.png', resize_2_color)
print('key points oct 2 in set 2')
for i in range(1,octave3_key02.shape[0]):
    for j in range(1,octave3_key02.shape[1]):
        if(octave3_key02[i][j]>0.9):
            resize_2_color[i][j]=255;
            print('x2,y2', i, j)
cv2.imshow("octave3 merge Image Key detection_2",resize_2_color)

print("Image written to file-system : ", status)
status = cv2.imwrite('G:\CVIP Project//octave3_key02.png', resize_2_color)
print("Image written to file-system : ", status)

#octave 4 Blur 1 image
g4_1=Gaussian(4 * math.sqrt(2))
conv4_1=convolution(g4_1, resize_3)
# Eliminate zero values with method 1
blur4_1 = (conv4_1 - np.min(conv4_1)) / (np.max(conv4_1) - np.min(conv4_1))
#cv2.namedWindow('Blurred Image octave 3_1', cv2.WINDOW_NORMAL)
cv2.imshow('octave4 blur4_1', blur4_1)


#octave 4 Blur 2 image
g4_2=Gaussian(8)
conv4_2=convolution(g4_2, resize_3)
# Eliminate zero values with method 1
blur4_2 = (conv4_2 - np.min(conv4_2)) / (np.max(conv4_2) - np.min(conv4_2))
#cv2.namedWindow('Blurred Image octave 3_1', cv2.WINDOW_NORMAL)
cv2.imshow('octave4 blur4_2', blur4_2)

#octave 4 Blur 3 image
g4_3=Gaussian(8*math.sqrt(2))
conv4_3=convolution(g4_3,resize_3)
# Eliminate zero values with method 1
blur4_3 = (conv4_3 - np.min(conv4_3)) / (np.max(conv4_3) - np.min(conv4_3))
#cv2.namedWindow('Blurred Image octave 3_1', cv2.WINDOW_NORMAL)
cv2.imshow('octave4 blur4_3', blur4_3)

#octave 4 Blur 4 image
g4_4=Gaussian(16)
conv4_4=convolution(g4_4,resize_3)
# Eliminate zero values with method 1
blur4_4 = (conv4_4 - np.min(conv4_4)) / (np.max(conv4_4) - np.min(conv4_4))
#cv2.namedWindow('Blurred Image octave 3_1', cv2.WINDOW_NORMAL)
cv2.imshow('octave4 blur4_3', blur4_4)

#octave 4 Blur 5 image
g4_5=Gaussian(16*math.sqrt(2))
conv4_5=convolution(g4_3,resize_3)
# Eliminate zero values with method 1
blur4_5 = (conv4_5 - np.min(conv4_5)) / (np.max(conv4_5) - np.min(conv4_5))
#cv2.namedWindow('Blurred Image octave 3_1', cv2.WINDOW_NORMAL)
cv2.imshow('octave4 blur4_3', blur4_5)


#octave4 DOG1
OCT4_DOG_Change_1=blur4_2-blur4_1
# Eliminate zero values with method 1
OCT4_DOG1 = (OCT4_DOG_Change_1 - np.min(OCT4_DOG_Change_1)) / (np.max(OCT4_DOG_Change_1) - np.min(OCT4_DOG_Change_1))
#cv2.namedWindow('DOG Image 1', cv2.WINDOW_NORMAL)
cv2.imshow('octave 4 DOG 1', OCT4_DOG1)

#octave4 DOG2
OCT4_DOG_Change_2=blur4_3-blur4_2
# Eliminate zero values with method 1
OCT4_DOG2 = (OCT4_DOG_Change_2 - np.min(OCT4_DOG_Change_2)) / (np.max(OCT4_DOG_Change_2) - np.min(OCT4_DOG_Change_2))
#cv2.namedWindow('DOG Image 2', cv2.WINDOW_NORMAL)
cv2.imshow(' octave 4 DOG 2', OCT4_DOG2)

#octave4 DOG3
OCT4_DOG_Change_3=blur4_4-blur4_3
# Eliminate zero values with method 1
OCT4_DOG3 = (OCT4_DOG_Change_3 - np.min(OCT4_DOG_Change_3)) / (np.max(OCT4_DOG_Change_3) - np.min(OCT4_DOG_Change_3))
#cv2.namedWindow('DOG Image 3', cv2.WINDOW_NORMAL)
cv2.imshow(' octave 4 DOG 3', OCT4_DOG3)

#octave4 DOG4
OCT4_DOG_Change_4=blur4_5-blur4_4
# Eliminate zero values with method 1
OCT4_DOG4 = (OCT4_DOG_Change_4 - np.min(OCT4_DOG_Change_4)) / (np.max(OCT4_DOG_Change_4) - np.min(OCT4_DOG_Change_4))
#cv2.imwrite("OCT1_DOG4 Write",OCT1_DOG4)
cv2.imshow(' octave 4 DOG 4', OCT4_DOG4)


#octave4 key detection
octave4_key1=np.zeros(OCT4_DOG3.shape);
octave4_key2=np.zeros(OCT4_DOG3.shape);
octave4_key1=keypoint1(OCT4_DOG1,OCT4_DOG2,OCT4_DOG3)
octave4_key2=keypoint2(OCT4_DOG2,OCT4_DOG3,OCT4_DOG4)
octave4_key01 = (octave4_key1 - np.min(octave4_key1)) / (np.max(octave4_key1) - np.min(octave4_key1))
octave4_key02 = (octave4_key2 - np.min(octave4_key2)) / (np.max(octave4_key2) - np.min(octave4_key2))

cv2.imshow("octave4 Image Key detection_1",octave4_key01)
cv2.imshow("octave4 Image Key detection_2",octave4_key02)


for i in range(1,octave4_key01.shape[0]):
    for j in range(1,octave4_key01.shape[1]):
        if(octave4_key01[i][j]>0.9):
            resize_3_color[i][j]=255;

cv2.imshow("octave4 merge Image Key detection_1",resize_3_color)

for i in range(1,octave4_key02.shape[0]):
    for j in range(1,octave4_key02.shape[1]):
        if(octave4_key01[i][j]>0.9):
            resize_3_color[i][j]=255;

cv2.imshow("octave4 merge Image Key detection_2",resize_3_color)

cv2.waitKey(0)
cv2.destroyAllWindows()
