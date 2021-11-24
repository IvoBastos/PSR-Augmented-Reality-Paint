#!/usr/bin/python3

# experimentar  ./desenho.py -d teste1.png

import argparse
import cv2
import numpy as np


def main():
    """
    INITIALIZE -----------------------------------------
    """
    parser = argparse.ArgumentParser(description="teste desenhos")
    parser.add_argument("-d", "--des", type=str, required=True, help="desenho a tratar")
    args = vars(parser.parse_args())

    image = cv2.imread(args["des"], cv2.IMREAD_COLOR)

    imagehsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imshow('original', image)  # Display the image

    height, width, _ = image.shape
    image_canvas = np.zeros((height, width), np.uint8)

    # print( " height, width 1::" +str(height) + " " + str(width))

    i = 0
    while True:
        if i == 4:
            break

        if i == 3:  # apanha o preto
            lower = np.array([0, 0, 0])
            upper = np.array([50, 50, 50])
            mask = cv2.inRange(imagehsv, lower, upper)

        if i == 0:  # apanha o verde
            lower = np.array([35, 150, 20])
            upper = np.array([70, 255, 255])
            mask = cv2.inRange(imagehsv, lower, upper)

        if i == 1:  # apanha o azul
            lower = np.array([70, 150, 20])
            upper = np.array([130, 255, 255])
            mask = cv2.inRange(imagehsv, lower, upper)

        if i == 2:  # apanha o vermelho
            # lower mask (0-10)
            lower_red = np.array([0, 50, 50])
            upper_red = np.array([10, 255, 255])
            mask0 = cv2.inRange(imagehsv, lower_red, upper_red)

            # upper mask (170-180)
            lower_red = np.array([170, 50, 50])
            upper_red = np.array([180, 255, 255])
            mask1 = cv2.inRange(imagehsv, lower_red, upper_red)

            # join my masks
            mask = mask0 + mask1

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        maskr = np.zeros(mask.shape, np.uint8)
        maskr2 = np.zeros(mask.shape, np.uint8)

        cv2.drawContours(maskr, contours, -1, 255, 1)

        # Preciso de uma imagem de trabalho onde possa preencher achar e achar o centro para depois escrever na mascara inicial
        cv2.drawContours(maskr2, contours, -1, (255), -1)

        # cv2.imshow('maskr2', maskr2)

        kernel = np.ones((3, 3), 'uint8')

        image_1 = cv2.erode(maskr2, kernel, iterations=3)
        #   image_1 = cv2.morphologyEx(image_1, cv2.MORPH_CLOSE, kernel)

        #  cv2.imshow('image_1', image_1)

        # contoursm, _ = cv2.findContours(image_1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # img_output, contoursm, hierarchy = cv2.findContours(image_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contoursm, _ = cv2.findContours(image_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contoursmi in contoursm:
            M = cv2.moments(contoursmi)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.putText(image_canvas, str(i + 1), (cx + 2, cy + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255),
                            2)

        image_canvas = cv2.add(image_canvas, maskr)

        # cv2.imshow('maskr', maskr)  # Display the image
        # cv2.imshow('image_canvas', image_canvas)

        # cv2.waitKey(0)

        i = i + 1
        # print(i)

    image_canvas = cv2.bitwise_not(image_canvas)
    cv2.imshow('pinta', image_canvas)  # Display the image
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
