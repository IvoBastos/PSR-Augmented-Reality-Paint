#!/usr/bin/python3


import argparse
import json
import time
import cv2
import numpy as np

# dictionary with ranges
ranges_pcss = {"b": {"min": 100, "max": 256},
               "g": {"min": 100, "max": 256},
               "r": {"min": 100, "max": 256},
               }


def main():
    """
    INITIALIZE -----------------------------------------
    """
    # program flags
    background_white = True  # background color
    pointer_on = False  # pointer method incomplete
    rect_drawing = False  # rectangle drawing flag
    circle_drawing = False  # circle drawing flag

    # variables
    dot_x, dot_y = 0, 0  # pen points
    prev_x, prev_y = 0, 0  # point for continuous draw
    rect_pt1_x, rect_pt1_y, rect_pt2_x, rect_pt2_y = 0, 0, 0, 0  # rectangle drawing points
    circle_pt1_x, circle_pt1_y, circle_pt2_x, circle_pt2_y = 0, 0, 0, 0  # circle drawing points

    # parse the json file with BGR limits (from color_segmenter.py)
    parser = argparse.ArgumentParser(description="Load a json file with limits")
    parser.add_argument("-j", "--json", type=str, required=True, help="Full path to json file")
    args = vars(parser.parse_args())

    # read the json file
    with open(args["json"], "r") as file_handle:
        data = json.load(file_handle)

    # print json file then close
    # print(data)  # debug
    file_handle.close()

    ranges_pcss["b"]["min"] = data["b"]["min"]
    ranges_pcss["b"]["max"] = data["b"]["max"]
    ranges_pcss["g"]["min"] = data["g"]["min"]
    ranges_pcss["g"]["max"] = data["g"]["max"]
    ranges_pcss["r"]["min"] = data["r"]["min"]
    ranges_pcss["r"]["max"] = data["r"]["max"]

    # print(ranges_pcss)  # debug

    # numpy arrays
    mins_pcss = np.array([ranges_pcss['b']['min'], ranges_pcss['g']['min'], ranges_pcss['r']['min']])
    maxs_pcss = np.array([ranges_pcss['b']['max'], ranges_pcss['g']['max'], ranges_pcss['r']['max']])

    # initial setup
    capture = cv2.VideoCapture(0)  # connect to webcam

    # create the window
    window_segmented = "Color Segmenter"
    cv2.namedWindow(window_segmented, cv2.WINDOW_AUTOSIZE)

    # create a white image background dim 600*400
    background = np.zeros((422, 750, 3), np.uint8)
    background.fill(255)

    # window for display background/draw area
    draw_area = "Draw Area"
    cv2.namedWindow(draw_area)

    # merged camera and drawing
    merged_area = "Interactive Drawing"
    cv2.namedWindow(merged_area)
    image_canvas = np.zeros((422, 750, 3), np.uint8)

    # pen variables
    pen_color = (0, 0, 0)
    pen_thickness = 5

    """
    EXECUTION -----------------------------------------
    """
    while True:
        # read the image
        _, image = capture.read()
        image = cv2.resize(image, (750, 422))  # resize the capture window

        # apply filters
        # blurred_frame = cv2.GaussianBlur(image, (5, 5), 0)

        # transform the image and show it
        mask = cv2.inRange(image, mins_pcss, maxs_pcss)  # colors mask
        image_segmenter = cv2.bitwise_and(image, image, mask=mask)

        # get contours
        contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # cv2.drawContours(image_segmenter, contours, -1, (0, 0, 255), 3)
        # cv2.drawContours(image, contours, -1, (0, 0, 255), 3)

        # create the rectangle over object and draw on the background
        for contours in contours:
            (x, y, w, h) = cv2.boundingRect(contours)

            # draw the rectangle only if the area is > 3000 px
            # print(cv2.contourArea(contours)) # debug
            if cv2.contourArea(contours) < 3000:
                continue

            # draw the rectangle
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # dot in the center of the rectangle
            dot_x = x + w / 2
            dot_y = y + h / 2
            cv2.circle(image, (int(dot_x), int(dot_y)), 10, (0, 0, 0), cv2.FILLED)

            # draw in the background
            if prev_x == 0 and prev_y == 0:  # skip first iteration
                prev_x, prev_y = dot_x, dot_y

            if abs(prev_x - dot_x) < 50 and abs(prev_y - dot_y) < 50:   # mitigate appears and disappears of the pen
                cv2.line(background, (int(prev_x), int(prev_y)), (int(dot_x), int(dot_y)), pen_color, pen_thickness)
                cv2.line(image_canvas, (int(prev_x), int(prev_y)), (int(dot_x), int(dot_y)), pen_color, pen_thickness)
                prev_x, prev_y = dot_x, dot_y
            else:
                prev_x, prev_y = 0, 0

            # --------------------------------
            # if not pointer_on:
            #     cv2.circle(background, (int(dot_x), int(dot_y)), pen_thickness, pen_color, cv2.FILLED)
            # else:
            #     cv2.circle(background, (int(dot_x), int(dot_y)), pen_thickness, pen_color, cv2.FILLED)
            #     # background.fill(255)


        # imshows
        cv2.imshow(window_segmented, image_segmenter)  # drawing object (pen)
        cv2.imshow(draw_area, background)  # draw area
        cv2.imshow("Original", image)

        # merge the video and the drawing ----------------------------INCOMPLETE DOESN'T DRAW THE BLACK COLOR
        # image_merged = cv2.addWeighted(image, 0.5, background, 0.5, 0) # merged the images to draw on the video
        # cv2.imshow(merged_area, image_merged)
        image_gray = cv2.cvtColor(image_canvas, cv2.COLOR_BGR2GRAY)
        _, image_inverse = cv2.threshold(image_gray, 50, 255, cv2.THRESH_BINARY_INV)
        image_inverse = cv2.cvtColor(image_inverse, cv2.COLOR_GRAY2BGR)
        image = cv2.bitwise_and(image, image_inverse)
        image = cv2.bitwise_or(image, image_canvas)
        cv2.imshow(merged_area, image)

        """
        interactive keys (k) -----------------------------------------
        """
        # ESC to close
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        # clean the screen
        if k == ord("c"):
            if background_white:
                background.fill(255)
                image_canvas.fill(0)
                print("CLEAN")
            else:
                background.fill(0)

        # red color
        if k == ord("r"):
            pen_color = (0, 0, 255)
            print("YOU SELECT RED COLOR")

        # green color
        if k == ord("g"):
            pen_color = (0, 255, 0)
            print("YOU SELECT GREEN COLOR")

        # blue color
        if k == ord("b"):
            pen_color = (255, 0, 0)
            print("YOU SELECT BLUE COLOR")

        # black color
        if k == ord("B"):
            if background_white:
                pen_color = (0, 0, 0)
                print("YOU SELECT BLACK COLOR")
            else:
                pen_color = (255, 255, 255)
                print("YOU SELECT WHITE COLOR")

        # thickness
        if k == ord("T"):
            pen_thickness += 1
            print("THICKNESS: " + str(pen_thickness))

        if k == ord("t"):
            pen_thickness -= 1
            if pen_thickness < 0:
                pen_thickness = 0
            print("THICKNESS: " + str(pen_thickness))

        # erase
        if k == ord("e"):
            if background_white:
                pen_color = (255, 255, 255)
                print("YOU SELECT ERASER")
            else:
                pen_color = (0, 0, 0)
                print("YOU SELECT ERASER")

        # flip the background
        if k == ord("f"):
            if background_white:
                background.fill(0)
                background_white = False
                pen_color = (255, 255, 255)
            else:
                background.fill(255)
                background_white = True
                pen_color = (0, 0, 0)

        # pointer mode ---- not working
        if k == ord("p"):
            if pointer_on:
                pointer_on = False
            else:
                pointer_on = True

        # draw a rectangle------------------------------------------ INCOMPLETE, DRAW MULTIPLE RECTANGLES
        if k == ord("R"):
            rect_drawing = True
            rect_pt1_x = int(dot_x)
            rect_pt1_y = int(dot_y)
            # print(rect_cnt)
            # print(rect_pt1_x, rect_pt1_y)

        if rect_drawing:
            rect_pt2_x = int(dot_x)
            rect_pt2_y = int(dot_y)
            cv2.rectangle(background, (rect_pt1_x, rect_pt1_y), (rect_pt2_x, rect_pt2_y), pen_color, cv2.FILLED)

        if k == ord("L") and rect_drawing:
            rect_pt2_x = int(dot_x)
            rect_pt2_y = int(dot_y)
            rect_drawing = False
            cv2.rectangle(background, (rect_pt1_x, rect_pt1_y), (rect_pt2_x, rect_pt2_y), pen_color, cv2.FILLED)

        # draw a circle------------------------------------------ INCOMPLETE, DOESN'T WORK PROPERLY
        if k == ord("C"):
            circle_drawing = True
            circle_pt1_x = int(dot_x)
            circle_pt1_y = int(dot_y)

        if circle_drawing:
            circle_pt2_x = int(dot_x)
            circle_pt2_y = int(dot_y)
            # cv2.circle(image, center_coordinates, radius, color, thickness)
            cv2.ellipse(background, (circle_pt1_x, circle_pt1_y),
                        (circle_pt2_x, circle_pt2_y), 45, 45, 360, pen_color, cv2.FILLED)

        if k == ord("L") and circle_drawing:
            circle_pt2_x = int(dot_x)
            circle_pt2_y = int(dot_y)
            circle_drawing = False
            cv2.ellipse(background, (circle_pt1_x, circle_pt1_y),
                        (circle_pt2_x, circle_pt2_y), 0, 0, 360, pen_color, cv2.FILLED)

    """
    FINALIZATION -----------------------------------------
    """
    capture.release()  # free the webcam for other use
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
