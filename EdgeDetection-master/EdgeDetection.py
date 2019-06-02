import cv2
import numpy as np

image=cv2.imread("G:\TestImage.png",0)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('test',image)

image_r=image.shape[0]  #Row
image_c=image.shape[1]  #Column

sobelx=[[-1,0,1],[-2,0,2],[-1,0,1]]         #sobel Horizontal operation
sobely=[[-1,-2,-1],[0 ,0 ,0],[1,2,1]]       #Sobel Vertical Operation

edge_x=np.zeros(image.shape)
edge_y=np.zeros(image.shape)

h=3
w=3
sumx=0
sumy=0
for i in range(1, image_r-1):
    for j in range(1, image_c-1):
        #Calculate sumx and sumy using Sobel (horizontal and vertical gradients)
        sumx = (sobelx[0][0] * image[i-1][j-1]) + (sobelx[0][1] * image[i-1][j]) + \
             (sobelx[0][2] * image[i-1][j+1]) + (sobelx[1][0] * image[i][j-1]) + \
             (sobelx[1][1] * image[i][j]) + (sobelx[1][2] * image[i][j+1]) + \
             (sobelx[2][0] * image[i+1][j-1]) + (sobelx[2][1] * image[i+1][j]) + \
             (sobelx[2][2] * image[i+1][j+1])
        sumy=(sobely[0][0] * image[i-1][j-1]) + (sobely[0][1] * image[i-1][j]) + \
             (sobely[0][2] * image[i-1][j+1]) + (sobely[1][0] * image[i][j-1]) + \
             (sobely[1][1] * image[i][j]) + (sobely[1][2] * image[i][j+1]) + \
             (sobely[2][0] * image[i+1][j-1]) + (sobely[2][1] * image[i+1][j]) + \
             (sobely[2][2] * image[i+1][j+1])
        edge_x[i][j] = sumx
        edge_y[i][j]   =sumy

# Eliminate zero values with method 2
pos_edge_x = np.abs(edge_x) / np.max(np.abs(edge_x))
cv2.namedWindow('pos_edge_x_dir', cv2.WINDOW_NORMAL)
cv2.imshow('pos_edge_x_dir', pos_edge_x)
status = cv2.imwrite('G:\CVIP Project//pos_edge_x_dir.png', 255*pos_edge_x)

# Eliminate zero values with method 2
pos_edge_y = np.abs(edge_y) / np.max(np.abs(edge_y))
cv2.namedWindow('pos_edge_y_dir', cv2.WINDOW_NORMAL)
cv2.imshow('pos_edge_y_dir', pos_edge_y)
status = cv2.imwrite('G:\CVIP Project//pos_edge_y_dir.png', 255*pos_edge_y)

# magnitude of edges (conbining horizontal and vertical edges)
edge_magnitude = np.sqrt(edge_x ** 2 + edge_y ** 2)
edge_magnitude /= np.max(edge_magnitude)
cv2.namedWindow('edge_magnitude', cv2.WINDOW_NORMAL)
cv2.imshow('edge_magnitude', edge_magnitude)

edge_direction = np.arctan(edge_y / (edge_x + 1e-3))
edge_direction = edge_direction * 180. / np.pi
edge_direction /= np.max(edge_direction)
cv2.namedWindow('edge_direction', cv2.WINDOW_NORMAL)
cv2.imshow('edge_direction', edge_magnitude)
status = cv2.imwrite('G:\CVIP Project//edge_magnitude.png', 255*edge_magnitude)
cv2.waitKey()
cv2.destroyAllWindows()

print("Original image size: {:4d} x {:4d}".format(image.shape[0], image.shape[1]))
print("Resulting image size: {:4d} x {:4d}".format(edge_magnitude.shape[0], edge_magnitude.shape[1]))