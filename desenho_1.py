#!/usr/bin/python3

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cv2 import FONT_ITALIC
from cv2.cv2 import LINE_8

def main():
    areaa_total_blob = 0

    ranges = {'b': {'min': 0, 'max': 256},
              'g': {'min': 0, 'max': 256},
              'r': {'min': 0, 'max': 256}}

    # Processing
    mins = np.array([ranges['b']['min'], ranges['g']['min'], ranges['r']['min']])
    maxs = np.array([ranges['b']['max'], ranges['g']['max'], ranges['r']['max']])


    image = cv2.imread('./BLOB6_0.png', cv2.IMREAD_COLOR)
    mascaras=[]

    # Base de dados das cores a Utilizar
    color_data_base=[(255, 0, 0),
                    (0, 255, 0),
                    (0, 0, 255)]

    #Chaves para avaliação das Imagens
    keys_of_Bloobs= {'BLOB5_02.png':(1,2,2,3,2,1,1,1,3),
                     'BLOB3_02.png':(1,2,3,3,1),
                     'BLOB4_02.png':(3,3,1,1,2,3,3,3,1,1,2,1,2,2),
                     'BLOB6_02.png':(1,3,3,3,1,1,2,2,2,2,2,3,3)}

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

        width = w > 10
        heigh = h > 10
        area_i = area > 20

        if all ((width, heigh, area_i)):

            componentmask=(labels == i).astype("uint8") * 255
            mascaras.append(cv2.bitwise_or(mask_3, componentmask))


    image_a_comparar= cv2.imread('./BLOB6_02.png', cv2.IMREAD_COLOR)
    mask_a_comparar = cv2.cvtColor(image_a_comparar, cv2.COLOR_BGR2GRAY)

    cv2.imshow('a_comparar', image_a_comparar)

    for i in range(0,len(mascaras)):

        # print(stats[i, cv2.CC_STAT_AREA])
        cv2.imshow('mask_' + str(i), mascaras[i])
        # print(np.array(color_data_base[(keys_of_Bloobs.get('BLOB6_02.png')[i])-1]))
        mask_range = cv2.inRange(image_a_comparar, mins, np.array(color_data_base[(keys_of_Bloobs.get('BLOB6_02.png')[i])-1]))
        mask_NEW = cv2.bitwise_and(mascaras[i], mask_range)

        output2 = cv2.connectedComponentsWithStats(mask_NEW, 8, cv2.CV_32S)
        (numLabels_1, labels_1, stats_1, centroids_1) = output2

        for a in range(1, numLabels_1):
            area_1 = stats_1[a, cv2.CC_STAT_AREA]
            areaa_total_blob += area_1
        # print(areaa_total_blob)
        # cv2.putText(mask_NEW, str(areaa_total_blob), (int(cX), int(cY)), FONT_ITALIC, 1, (0, 0, 255), 2, LINE_8)
        cv2.imshow('mask_NEW' + str(i), mask_NEW)



    cv2.waitKey(-1)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


