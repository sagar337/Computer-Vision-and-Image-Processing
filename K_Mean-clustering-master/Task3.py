import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt

x = [[5.9, 3.2], [4.6, 2.9], [6.2, 2.8], [4.7, 3.2], [5.5, 4.2], [5.0, 3.0], [4.9, 3.1], [6.7, 3.1], [5.1, 3.8],
     [6.0, 3.0]]
N = 10
#     ######RED########GREEN##########BLUE######
original_means = [[6.2, 3.2], [6.6, 3.7], [6.5, 3.0]]


def euclidean_distance(p1, p2):
    return (((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2)) ** 0.5


def cluster_mean(cluster):
    sum_x = 0
    sum_y = 0
    n = len(cluster)
    for point in cluster:
        sum_x += point[0]
        sum_y += point[1]
    return [round(sum_x / n, 4), round(sum_y / n, 4)]


def plot_cluster_points(cluster, color):
    for point in cluster:
        plt.scatter(point[0], point[1], marker='^', c=color)
        plt.text(point[0], point[1], "({},{})".format(point[0], point[1]))


def plot_mean(mean, color):
    plt.scatter(mean[0], mean[1], c=color)
    plt.text(mean[0], mean[1]-0.075, "(M:{},{})".format(mean[0], mean[1]))


def plot_points_with_means(means, clusters, file_name):
    plot_mean(means[0], 'red')
    plot_mean(means[1], 'green')
    plot_mean(means[2], 'blue')
    plot_cluster_points(clusters[0], 'red')
    plot_cluster_points(clusters[1], 'green')
    plot_cluster_points(clusters[2], 'blue')
    plt.savefig(file_name)
    plt.clf()


def perform_classification(input_points, k_means):
    classification = []
    cluster1 = []
    cluster2 = []
    cluster3 = []
    for point in input_points:
        d1 = euclidean_distance(point, k_means[0])
        d2 = euclidean_distance(point, k_means[1])
        d3 = euclidean_distance(point, k_means[2])
        cluster = 3
        if d1 < d2:
            if d1 < d3:
                cluster = 1
                cluster1.append(point)
            else:
                cluster3.append(point)
        else:
            if d2 < d3:
                cluster = 2
                cluster2.append(point)
            else:
                cluster3.append(point)
        classification.append(cluster)

    return {
        "classification_vector": classification,
        "clusters": [cluster1, cluster2, cluster3],
        "new_means": [cluster_mean(cluster1), cluster_mean(cluster2), cluster_mean(cluster3)]
    }



def print_classification_vector(vector):
    for i in range(len(vector)):
        print("Point {} is in cluster {}".format(str(x[i]), vector[i]))


result_of_first_clustering = perform_classification(x, original_means)

clusters = result_of_first_clustering["clusters"]
first_iteration_means = result_of_first_clustering["new_means"]

# This will solve assignment problem 3.1
plot_points_with_means(original_means, clusters, 'task3_iter1_a.jpg')
print_classification_vector(result_of_first_clustering["classification_vector"])
# This will solve assignment problem 3.2
plot_points_with_means(first_iteration_means, clusters, 'task3_iter1_b.jpg')
print()
# 2nd classification
result_of_second_clustering = perform_classification(x, first_iteration_means)
clusters = result_of_second_clustering["clusters"]
second_iteration_means = result_of_second_clustering["new_means"]

# This will solve assignment problem 3.3
print_classification_vector(result_of_second_clustering["classification_vector"])
plot_points_with_means(first_iteration_means, clusters, 'task3_iter2_a.jpg')
plot_points_with_means(second_iteration_means, clusters, 'task3_iter2_b.jpg')

#3.4
def perform_classification_2(input_points, k_means):
    classification = []
    cluster1 = []
    cluster2 = []
    cluster3 = []
    for i in range(1,input_points.shape[0]-1):
        for j in range(1,input_points.shape[1]-1):
            d1 = euclidean_distance(input_points[i][j], k_means[0])
            d2 = euclidean_distance(input_points[i][j], k_means[1])
            d3 = euclidean_distance(input_points[i][j], k_means[2])
            cluster = 3
            if d1 < d2:
                if d1 < d3:
                    cluster = 1
                    cluster1.append(input_points[i][j])
                else:
                    cluster3.append(input_points[i][j])
            else:
                if d2 < d3:
                    cluster = 2
                    cluster2.append(input_points[i][j])
                else:
                    cluster3.append(input_points[i][j])
            classification.append(cluster)
            print(cluster)

    return {
        "classification_vector": classification,
        "clusters": [cluster1, cluster2, cluster3],
        "new_means": [cluster_mean(cluster1), cluster_mean(cluster2), cluster_mean(cluster3)]
    }

img=cv2.imread('baboon.jpg')
Test_img=img.reshape((-1,3))
Test_img=np.float32(Test_img)

#Number of clusters k=3
k=3
C_x=np.random.randint(0, np.max(Test_img)-20, size=k)
C_y=np.random.randint(0, np.max(Test_img)-20, size=k)
C_z=np.random.randint(0, np.max(Test_img)-20, size=k)

org_means=[C_x,C_y,C_z]
a = np.array(img)
result_of_first_clustering = perform_classification_2(img, org_means)
print("result of 1st clustering")
print(result_of_first_clustering)
clusters = result_of_first_clustering["clusters"]
first_iteration_means = result_of_first_clustering["new_means"]
plot_points_with_means(org_means, clusters, 'task3_iter3_a.jpg')
print_classification_vector(result_of_first_clustering["classification_vector"])
center = np.uint8(first_iteration_means)
res = center[print_classification_vector.flatten()]
res2 = res.reshape((img.shape))

cv2.imshow('res2',res2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# This will solve assignment problem 3.1
plot_points_with_means(org_means, clusters, 'task3_iter3_a.jpg')
print("classfication vector for 3.1:")
print_classification_vector(result_of_first_clustering["classification_vector"])
print("Mean after 1st iteration for 3.1:")
print(result_of_first_clustering["new_means"])
# This will solve assignment problem 3.2
plot_points_with_means(first_iteration_means, clusters, 'task3_iter3_b.jpg')

result_of_second_clustering = perform_classification_2(img, first_iteration_means)
print("classfication vector for 3.2:")
print_classification_vector(result_of_second_clustering["classification_vector"])
clusters = result_of_second_clustering["clusters"]
print("Mean after 2nd iteration for 3.2:")
print(result_of_second_clustering["new_means"])
second_iteration_means = result_of_second_clustering["new_means"]

print()

def rgb_euclidean_distance(p1, p2):
    return (((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2) + ((p1[2] - p2[2]) ** 2)) ** 0.5

def cluster_mean(cluster):
    sum_r = 0
    sum_g = 0
    sum_b = 0
    n = len(cluster)
    if n is 0:
        return [0, 0, 0]
    for point in cluster:
        sum_r += point[0]
        sum_g += point[1]
        sum_b += point[2]
    print(round(sum_r / n, 4), round(sum_g / n, 4), round(sum_b / n, 4))
    print()
    return [round(sum_r / n, 4), round(sum_g / n, 4), round(sum_b / n, 4)]

def compare_means(mean1, mean2):
    if len(mean1) == len(mean2):
        for m1, m2 in zip(mean1, mean2):
            if m1[0] != m2[0] or m1[1] != m2[1] or m1[2] != m2[2]:
                return False
    else:
        return False
    return True


def segment_image(image_as_array, initial_means, iterations):
    old_means = np.copy(initial_means)
    segmentation_result = {}
    for i in range(iterations):
        print(" iteration number : " + str(i))
        segmentation_result = perform_classification(image_as_array, initial_means)
        if compare_means(old_means, segmentation_result["new_means"]):
            break
    return segmentation_result


def perform_classification(input_points, k_means):
    classification = []
    clusters = [[] for j in range(len(k_means))]
    for point in input_points:
        closest_mean = 0
        distance_to_closest_mean = rgb_euclidean_distance(point, k_means[0])
        for i in range(1, len(k_means)):
            print(i)
            distance_to_current_mean = rgb_euclidean_distance(point, k_means[i])
            if distance_to_current_mean < distance_to_closest_mean:
                closest_mean = i
                distance_to_closest_mean = distance_to_current_mean
        clusters[closest_mean].append(point)
        classification.append(closest_mean)

    new_means = []
    for cluster in clusters:
        new_means.append(cluster_mean(cluster))

    return {
        "classification_vector": classification,
        "clusters": clusters,
        "new_means": new_means
    }
image = cv2.imread('baboon.jpg')
###########################Update k and Iteration as per requireemnt###############################
k = 20 # k can be 3,5,10,20
iterations = 10  # Iteration can be 5,10,20
initial_means = []

image_width = image.shape[0]
image_height = image.shape[1]

image_as_array = np.reshape(image, [image_width * image_height, 3])

for i in range(k):
    x = min(np.math.floor((image_width / (k - 1)) * i), image_width-1)
    y = min(np.math.floor((image_height / (k - 1)) * i), image_height-1)
    initial_means.append(image[x][y])

result_of_segmentation = segment_image(image_as_array, initial_means, iterations)

color_segmented_image = map(lambda x: result_of_segmentation["new_means"][x],
                            result_of_segmentation["classification_vector"])
color_segmented_image = np.reshape(list(color_segmented_image), [image_width, img])
cv2.imwrite('task3_baboon_20_10itr.jpg', color_segmented_image)