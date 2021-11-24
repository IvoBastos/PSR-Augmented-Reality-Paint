#!/usr/bin/python3

from pprint import pprint
import cv2
import numpy as np
from numpy import ndarray


def paint_evaluation(init_image_name):
    mascaras = []
    total_areas_blobs = []
    painted_areas_blobs = []
    accuracy = []
    results = {}

    ranges = {'b': {'min': 0, 'max': 256},
              'g': {'min': 0, 'max': 256},
              'r': {'min': 0, 'max': 256}}

    # Processing
    mins: ndarray = np.array([ranges['b']['min'], ranges['g']['min'], ranges['r']['min']])

    compare_name = str(init_image_name) + '4.png'
    image_name = str(init_image_name) + '.png'

    image = cv2.imread(image_name, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (864, 486))

    # Data base of the colors
    color_data_base = [(255, 0, 0),
                       (0, 255, 0),
                       (0, 0, 255)]

    # keys to evaluate the images
    keys_of_blobs = {'BLOB5_04.png': (1, 2, 2, 3, 2, 1, 1, 1, 3),
                     'BLOB3_04.png': (1, 2, 3, 3, 1),
                     'BLOB4_04.png': (3, 3, 1, 1, 2, 3, 3, 3, 1, 1, 2, 1, 2, 2),
                     'BLOB6_04.png': (1, 3, 3, 3, 1, 1, 2, 2, 2, 2, 2, 3, 3)}

    mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask_3 = np.zeros(mask.shape, dtype="uint8")

    # Applying command to find properties
    output = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output

    for i in range(1, numLabels):

        components = (labels == i).astype("uint8") * 255
        mascaras.append(cv2.bitwise_or(mask_3, components))

    compare_image = cv2.imread(compare_name, cv2.IMREAD_COLOR)
    compare_image = cv2.resize(compare_image, (864, 486))

    for i in range(0, len(mascaras)):

        mask_range = cv2.inRange(compare_image, mins,
                                 np.array(color_data_base[(keys_of_blobs.get(compare_name)[i]) - 1]))

        matching_of_mask = cv2.bitwise_and(mascaras[i], mask_range)

        # After many hours wasted the result of the image area continued to give very high values
        # which led to thinking that the image had to be inverted as a solution, it was necessary to do '255-mask_NEW'

        output2 = cv2.connectedComponentsWithStats((255 - matching_of_mask), 8, cv2.CV_32S)
        (numLabels_2, labels_2, stats_2, centroids_2) = output2

        painted_areas_blobs.append(stats_2[0, cv2.CC_STAT_AREA] / 255)
        total_areas_blobs.append(stats[i + 1, cv2.CC_STAT_AREA] / 255)

        accuracy.append((painted_areas_blobs[i] / total_areas_blobs[i]) * 100)

    results.update({'Number of areas': len(mascaras),
                    'Areas ta painter': total_areas_blobs,
                    'Areas painted': painted_areas_blobs,
                    'accuracy': accuracy
                    })
    print('')
    pprint(results)
    # cv2.waitKey(-1)
    cv2.destroyAllWindows()
