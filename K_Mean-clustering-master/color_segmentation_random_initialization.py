import matplotlib.pyplot as plt
import cv2
import numpy as np
from matplotlib import pyplot as plt


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


k = 3
iterations = 20
c_x = np.random.randint(0, 255, size=k)
c_y = np.random.randint(0, 255, size=k)
c_z = np.random.randint(0, 255, size=k)

initial_means = np.column_stack((c_x, c_y, c_z))

image_width = image.shape[0]
image_height = image.shape[1]

image_as_array = np.reshape(image, [image_width * image_height, 3])

result_of_segmentation = segment_image(image_as_array, initial_means, iterations)

color_segmented_image = map(lambda x: result_of_segmentation["new_means"][x],
                            result_of_segmentation["classification_vector"])

color_segmented_image = np.reshape(list(color_segmented_image), [image_width, image_height, 3])
cv2.imwrite('color_segmented_image.jpg', color_segmented_image)

