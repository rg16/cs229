from __future__ import division, print_function
import numpy as np
import random
import fp.load as load
import argparse
from pathlib import Path
import csv
import cv2

NUM_CLUSTERS = 3
COLOR_MAP = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255)
]

SAMPLES = [
    "CSC070EY-HT00-06_Split_Trans",
    "CSC084AQ-HT00-03_Split_Trans"
]


def init_centroids(num_clusters, data):
    """
    Initialize a `num_clusters` x image_shape[-1] nparray to RGB
    values of randomly chosen pixels of`image`

    Parameters
    ----------
    num_clusters : int
        Number of centroids/clusters
    data : nparray
        (H, W, C) image represented as an nparray

    Returns
    -------
    centroids_init : nparray
        Randomly initialized centroids
    """

    # *** START YOUR CODE ***

    centroids_init = np.asarray(random.sample(list(data), num_clusters))

    # *** END YOUR CODE ***

    return centroids_init


def get_closest_centroid(centroids, x):

    return np.argmin(np.linalg.norm(centroids-x, axis=1))


def classify(centroids, data):
    classes = -np.ones(data.shape[:-1])

    for c in range(len(data)):
        classes[c] = get_closest_centroid(centroids, data[c, :])
    return classes


def update_from_classes(data, classes, num_classes):
    centroids = np.zeros((num_classes, data.shape[-1]))
    for i in range(num_classes):
        centroids[i] = np.mean(data[classes == i], axis=0)
    return centroids


def update_centroids(centroids, data, max_iter=30, print_every=10):
    """
    Carry out k-means centroid update step `max_iter` times

    Parameters
    ----------
    centroids : nparray
        The centroids stored as an nparray
    image : nparray
        (H, W, C) image represented as an nparray
    max_iter : int
        Number of iterations to run
    print_every : int
        Frequency of status update

    Returns
    -------
    new_centroids : nparray
        Updated centroids
    """

    for i in range(max_iter):
        if i % print_every == 0:
            print(f"Beginning iteration {i}")

        # First, we assign each pixel to a class
        classes = classify(centroids, data)

        # Then, we use the classes to update our centroids

        new_centroids = update_from_classes(data, classes, centroids.shape[0])

        if np.linalg.norm(new_centroids - centroids) < 1e-5:
            break

        centroids = new_centroids

    return new_centroids


def find_closest(centroids, data):

    return np.argmin(np.linalg.norm(centroids - data, axis=1))


def run_kmeans(args):

    data = load.load_defect_data()
    print("Initializing Centroids...")
    centroids_init = init_centroids(args.clusters, data)

    print("Updating Centroids...")
    centroids = update_centroids(centroids_init, data, args.max_iter, 10)

    csv_paths = []
    for sample in SAMPLES:
        csv_paths.append(Path.cwd() / f"final-project/fa-tears/{sample}_defects.csv") # noqa
    for i, csv_path in enumerate(csv_paths):
        image_name = str(csv_path).split("_defects")[0] + ".png"
        image = cv2.imread(image_name)
        with csv_path.open("r") as csvfile:
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                defect = load.Defect.from_csv_row(row)
                vec = np.asarray([defect.max_int, defect.area])
                centroid = find_closest(centroids, vec)
                cv2.rectangle(
                    image,
                    (defect.left-100, defect.top-100),
                    (defect.left + defect.width+100, defect.top + defect.height + 100),
                    COLOR_MAP[centroid],
                    thickness=10
                )
        cv2.imwrite(f"k-means-{i}.png", image)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--max_iter', type=int, default=150,
                        help='Maximum number of iterations')
    parser.add_argument('--clusters', type=int, default=3,
                        help='Number of centroids/clusters')
    args = parser.parse_args()
    run_kmeans(args)

