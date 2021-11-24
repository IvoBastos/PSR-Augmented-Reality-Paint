#!/usr/bin/python3

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cv2 import FONT_ITALIC
from cv2.cv2 import LINE_8

def main():
    ranges = {'b': {'min': 0, 'max': 256},
              'g': {'min': 0, 'max': 256},
              'r': {'min': 0, 'max': 256}}

    # Processing
    mins = np.array([ranges['b']['min'], ranges['g']['min'], ranges['r']['min']])
    maxs = np.array([ranges['b']['max'], ranges['g']['max'], ranges['r']['max']])


    image = cv2.imread('./BLOB3.png', cv2.IMREAD_COLOR)
    image = cv2.resize(image, (750, 422))
    mascaras=[]

    color_data_base=[(255, 0, 0),
                    (0, 255, 0),
                    (0, 0, 255)]

    keys_of_Bloobs= {'BLOB6.png':(1,2,2,3,2,1,1,1,3)}

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

        # print(str(area))

        # cX, cY = int(cX), int(cY)
        cv2.putText(image, str(i), (int(cX), int(cY)), FONT_ITALIC, 1, (255, 0, 0), 2, LINE_8)
        # cv2.imshow('image', image)

        width = w > 100 and w < 1000
        heigh = h > 100 and h < 800
        area_i = area > 200

        if all ((width, heigh, area_i)):

            componentmask=(labels == i).astype("uint8") * 255
            mascaras.append(cv2.bitwise_or(mask_3, componentmask))


    image_a_comparar= cv2.imread('./BLOB5_02.png', cv2.IMREAD_COLOR)
    image_a_comparar = cv2.resize(image_a_comparar, (750, 422))
    mask_a_comparar = cv2.cvtColor(image_a_comparar, cv2.COLOR_BGR2GRAY)

    # cv2.imshow('a_comparar', image_a_comparar)
    final_frame_h1 = cv2.hconcat((image, image_a_comparar))

    for i in range(0,len(mascaras)):
        cv2.imshow('mask_' + str(i), mascaras[i])
        print(np.array(color_data_base[(keys_of_Bloobs.get('BLOB6.png')[i])-1]))
        mask_range = cv2.inRange(image_a_comparar, mins, np.array(color_data_base[(keys_of_Bloobs.get('BLOB6.png')[i])-1]))
        mask_NEW = cv2.bitwise_and(mascaras[i], mask_range)
        # mask_NEW2 = cv2.bitwise_xor(mascaras[i], mask_NEW)

        # cv2.imshow('mask_NEW' + str(i), mask_NEW)

        final_frame_h2 = cv2.hconcat((mascaras[i], mask_NEW))
        final_frame = cv2.vconcat((final_frame_h1,final_frame_h2))

        # Show the concatenated frame using imshow.
        cv2.imshow('frame'+ str(i),final_frame)


    cv2.waitKey(-1)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


