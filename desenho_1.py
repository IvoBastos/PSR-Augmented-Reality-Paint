#!/usr/bin/python3
from pprint import pprint

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cv2 import FONT_ITALIC
from cv2.cv2 import LINE_8

def Paint_avalue(Init_image_name):
    area_total_blob = 0
    Total_areas_Blobs = []
    Painted_areas_Blobs = []
    accuracy = []
    results = {}  # Analtino

    ranges = {'b': {'min': 0, 'max': 256},
              'g': {'min': 0, 'max': 256},
              'r': {'min': 0, 'max': 256}}

    # Processing
    mins = np.array([ranges['b']['min'], ranges['g']['min'], ranges['r']['min']])
    maxs = np.array([ranges['b']['max'], ranges['g']['max'], ranges['r']['max']])

    compare_name = str(Init_image_name) + '4.png'
    image_name = str(Init_image_name) + '.png'

    image = cv2.imread(image_name, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (750, 422))

    print(compare_name)
    mascaras=[]

    #Base de dados das cores a Utilizar
    color_data_base=[(255, 0, 0),
                    (0, 255, 0),
                    (0, 0, 255)]

    #Chaves para avaliação das Imagens
    keys_of_Bloobs= {'BLOB5_04.png':(1,2,2,3,2,1,1,1,3),
                     'BLOB3_04.png':(1,2,3,3,1),
                     'BLOB4_04.png':(3,3,1,1,2,3,3,3,1,1,2,1,2,2),
                     'BLOB6_04.png':(1,3,3,3,1,1,2,2,2,2,2,3,3)}

    mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask_3 = np.zeros(mask.shape, dtype="uint8")

    output = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output

    for i in range(1, numLabels):

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        (cX, cY) = centroids[i]

        cv2.putText(image, str(i), (int(cX), int(cY)), FONT_ITALIC, 1, (255, 0, 0), 2, LINE_8)
        cv2.imshow('image', image)

        width = w > 1
        heigh = h > 1
        area_i = area > 1

        if all ((width, heigh, area_i)):

            componentmask=(labels == i).astype("uint8") * 255
            mascaras.append(cv2.bitwise_or(mask_3, componentmask))

    image_a_comparar = cv2.imread(compare_name, cv2.IMREAD_COLOR)
    image_a_comparar = cv2.resize(image_a_comparar, (750, 422))
    # image_a_comparar= cv2.imread(teste, cv2.IMREAD_COLOR)
    # mask_a_comparar = cv2.cvtColor(image_a_comparar, cv2.COLOR_BGR2GRAY)

    cv2.imshow('a_comparar', image_a_comparar)

    for i in range(0,len(mascaras)):

        cv2.imshow('mask_' + str(i), mascaras[i])
        # print(np.array(color_data_base[(keys_of_Bloobs.get(compare_name)[i])-1]))
        mask_range = cv2.inRange(image_a_comparar, mins, np.array(color_data_base[(keys_of_Bloobs.get(compare_name)[i])-1]))

        mask_NEW = cv2.bitwise_and(mascaras[i], mask_range)

        output2 = cv2.connectedComponentsWithStats(mask_NEW, 8, cv2.CV_32S)
        (numLabels_1, labels_1, stats_1, centroids_1) = output2

        # for a in range(1, numLabels_1):
        #     area_1 = stats_1[a, cv2.CC_STAT_AREA]
        #     area_total_blob += area_1

        Total_areas_Blobs.append(stats[i, cv2.CC_STAT_AREA])
        Painted_areas_Blobs.append(stats_1[i, cv2.CC_STAT_AREA])
        accuracy.append((area_total_blob/stats[i, cv2.CC_STAT_AREA])*100)
        # print(area_total_blob)
        # cv2.putText(mask_NEW, str(areaa_total_blob), (int(cX), int(cY)), FONT_ITALIC, 1, (0, 0, 255), 2, LINE_8)
        cv2.imshow('mask_NEW' + str(i), mask_NEW)

    results.update({'Number of areas':numLabels_1,
                    'Areas a pintar':Total_areas_Blobs,
                    'Areas pintadas':Painted_areas_Blobs,
                    'accuracy': accuracy
                    })
    pprint(results)
    cv2.waitKey(-1)
    cv2.destroyAllWindows()



