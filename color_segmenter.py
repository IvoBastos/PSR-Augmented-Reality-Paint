#!/usr/bin/python3

import json
import cv2
import numpy as np

"""
Script to segment the colors on a video from the pc webcam
"""


def onTrackbar(threshold):
    pass


def main():
    # initial setup
    capture = cv2.VideoCapture(0)  # connect to webcam

    # create the window
    window_segmented = "Color Segmenter"
    window_regular = "Video Capture"
    cv2.namedWindow(window_segmented, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(window_regular, cv2.WINDOW_AUTOSIZE)

    # create the trackbars
    cv2.createTrackbar("min B", window_segmented, 0, 255, onTrackbar)
    cv2.createTrackbar("max B", window_segmented, 255, 255, onTrackbar)
    cv2.createTrackbar("min G", window_segmented, 0, 255, onTrackbar)
    cv2.createTrackbar("max G", window_segmented, 255, 255, onTrackbar)
    cv2.createTrackbar("min R", window_segmented, 0, 255, onTrackbar)
    cv2.createTrackbar("max R", window_segmented, 255, 255, onTrackbar)

    # dictionary with ranges
    ranges_pcss = {"b": {"min": 100, "max": 256},
                   "g": {"min": 100, "max": 256},
                   "r": {"min": 100, "max": 256},
                   }

    # cycle for continuously capture the image
    while True:

        # read the image
        _, image = capture.read()
        image = cv2.resize(image, (750, 422))  # resize the capture window

        # read the values on the trackbars
        min_b_pcss = cv2.getTrackbarPos("min B", window_segmented)
        max_b_pcss = cv2.getTrackbarPos("max B", window_segmented)
        min_g_pcss = cv2.getTrackbarPos("min G", window_segmented)
        max_g_pcss = cv2.getTrackbarPos("max G", window_segmented)
        min_r_pcss = cv2.getTrackbarPos("min R", window_segmented)
        max_r_pcss = cv2.getTrackbarPos("max R", window_segmented)

        # attribute the read values to the ranges dictionary
        ranges_pcss["b"]["min"] = min_b_pcss
        ranges_pcss["b"]["max"] = max_b_pcss
        ranges_pcss["g"]["min"] = min_g_pcss
        ranges_pcss["g"]["max"] = max_g_pcss
        ranges_pcss["r"]["min"] = min_r_pcss
        ranges_pcss["r"]["max"] = max_r_pcss

        # numpy arrays
        mins_pcss = np.array([ranges_pcss['b']['min'], ranges_pcss['g']['min'], ranges_pcss['r']['min']])
        maxs_pcss = np.array([ranges_pcss['b']['max'], ranges_pcss['g']['max'], ranges_pcss['r']['max']])

        # transform the image and show it
        mask = cv2.inRange(image, mins_pcss, maxs_pcss)  # colors mask
        image_segmenter = cv2.bitwise_and(image, image, mask=mask)


        # imshows
        cv2.imshow(window_segmented, image_segmenter)
        cv2.imshow(window_regular, image)  # regular camara

        """
        interactive keys (k) -----------------------------------------
        """
        # ESC to close
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        # w to save the json file
        if k == ord("w"):
            # save json file
            file_name = 'limits.json'
            with open(file_name, 'w') as file_handle:
                print('Limits file saved in directory by the name ' + file_name)
                print("Press q to quit or w to save a new file")
                json.dump(ranges_pcss, file_handle)

        # q to quit
        if k == ord("q"):
            break

    capture.release()  # free the webcam for other use
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
