# K_Mean-clustering
Clustering image coordinate into three cluster and color quantinzation of image


[5:9 3:2,
4:6 2:9,
6:2 2:8,
4:7 3:2,
5:5 4:2,
5:0 3:0,
4:9 3:1,
6:7 3:1,
5:1 3:8,
6:0 3:0]
Given the matrix X whose rows represent dierent data points, you are asked to perform a
k-means clustering on this dataset using the Euclidean distance as the distance function. Here k is
chosen as 3. All data in X were plotted in above Figure. The centers of 3 clusters were initialized
as 1 = (6:2; 3:2) (red), 2 = (6:6; 3:7) (green), 3 = (6:5; 3:0) (blue).
Implement the k-means clustering algorithm (you are only allowed to use the basic numpy
routines to implement the algorithm).
1. Classify N = 10 samples according to nearest i(i = 1; 2; 3). Plot the results by coloring the
empty triangles in red, blue or green. Include the classication vector and the classication
plot (task3 iter1 a.jpg) in the report. (1pt)
(a) [Hint:] Using plt.scatter with edgecolor, facecolor, marker and plt.text to plot the
gure.
2. Recompute i. Plot the updated i in solid circle in red, blue, and green respectively. Include
the updated i values and the plot in the report (task3 iter1 b.jpg). (1pt)
3. For a second iteration, plot the classication plot and updated i plot for the second iteration.
Include the classication vector and updated i values and these two plots (task3 iter2 a.jpg,
task3 iter2 b.jpg) in the report. (1pt)
4. [Color Quantization] Apply k-means to image color quantization. Using only k colors to
represent the image baboon.jpg. Include the color quantized images for k = 3; 5; 10; 20
(task3 baboon 3.jpg, task3 baboon 5.jpg, task3 baboon 10.jpg, task3 baboon 20.jpg).
(2pt)
