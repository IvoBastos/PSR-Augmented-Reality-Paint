#!/usr/bin/python3

import argparse
import copy
import json
import math
from functools import partial
import cv2
import numpy as np
from cv2 import FONT_ITALIC, LINE_8
from time import ctime
from colorama import Fore, Back, Style
from Paint_Avalue import paint_evaluation

# dictionary with BLOBs
Images_names = {"BLOB3_0.png": 'BLOB3_0',
                "BLOB4_0.png": 'BLOB4_0',
                "BLOB5_0.png": 'BLOB5_0',
                "BLOB6_0.png": 'BLOB6_0',
                }

# dictionary with range
ranges_pcss = {"b": {"min": 100, "max": 256},
               "g": {"min": 100, "max": 256},
               "r": {"min": 100, "max": 256},
               }

# global variables
drawing = False  # true in drawing shame mode
ix, iy, Drag = -1, -1, False  # used in def shape
pt1_y, pt1_x, moving_mouse = 0, 0, False  # used in def mouse_draw

# create a white image background
background = np.zeros((486, 864, 3), np.uint8)
background.fill(255)

# image load on parse -im
image_load = None
image_canvas = None


def shape(event, x, y, flags, params, mode, pen_color, pen_thickness):
    global ix, iy, Drag, background, image_canvas

    if event == cv2.EVENT_LBUTTONDOWN and not Drag:
        # value of variable draw will be set to True, when you press DOWN left mouse button
        # mouse location is captured here
        ix, iy = x, y
        Drag = True

    elif event == cv2.EVENT_MOUSEMOVE:
        # Dragging the mouse at this juncture

        if Drag:
            background = cv2.imread("temp.png")
            if mode == 'rectangle' and Drag:
                # If draw is True then it means you've clicked on the left mouse button
                # Here we will draw a rectangle from previous position to the x,y where the mouse is currently located
                cv2.rectangle(background, (ix, iy), (x, y), pen_color, pen_thickness)
                a = x
                b = y
                if a != x | b != y:
                    cv2.rectangle(background, (ix, iy), (x, y), pen_color, pen_thickness)

            if mode == 'circle':
                try:
                    radius = math.pow(((math.pow(x, 2) - math.pow(ix, 2)) + (math.pow(y, 2) - math.pow(iy, 2))), 1 / 2)
                    cv2.circle(background, (ix, iy), int(radius), pen_color, pen_thickness)
                    a = x
                    b = y
                    if a != x | b != y:
                        radius = math.pow(((math.pow(x, 2) - math.pow(ix, 2)) + (math.pow(y, 2) - math.pow(iy, 2))),
                                          1 / 2)
                        cv2.circle(background, (ix, iy), int(radius), pen_color, pen_thickness)
                except ValueError:
                    pass

    elif event == cv2.EVENT_LBUTTONDOWN and Drag:
        Drag = False

        if mode == 'rectangle' and not Drag:
            # As soon as you release the mouse button, variable draw will be set as False
            # Here we are completing to draw the rectangle on image window
            cv2.rectangle(background, (ix, iy), (x, y), pen_color, pen_thickness)
            cv2.rectangle(image_canvas, (ix, iy), (x, y), pen_color, pen_thickness)

        if mode == 'circle':
            radius = math.pow(((math.pow(x, 2) - math.pow(ix, 2)) + (math.pow(y, 2) - math.pow(iy, 2))), 1 / 2)
            cv2.circle(background, (ix, iy), int(radius), pen_color, pen_thickness)
            cv2.circle(image_canvas, (ix, iy), int(radius), pen_color, pen_thickness)

        cv2.imwrite('./temp' + '.png', background)  # Save the drawing for temp use
        background = cv2.imread("temp.png")

        return drawing


# paint with mouse on load image
def mouse_draw(event, x, y, flags, param, pen_color, pen_thickness):
    global pt1_y, pt1_x, image_load
    global moving_mouse

    # detect if left button is pressed
    if event == cv2.EVENT_LBUTTONDOWN:
        moving_mouse = True
        pt1_x, pt1_y = x, y
        # print(x, y)

    # disable the flag to stop drawing
    if event == cv2.EVENT_LBUTTONUP:
        moving_mouse = False
        cv2.line(image_load, (pt1_x, pt1_y), (x, y), pen_color, pen_thickness)

    # draw while mouse is moving
    if event == cv2.EVENT_MOUSEMOVE:
        if moving_mouse:
            cv2.line(image_load, (pt1_x, pt1_y), (x, y), pen_color, pen_thickness)
            pt1_x, pt1_y = x, y


def prepare_image(image_catch):
    image_ip = cv2.imread(image_catch, cv2.IMREAD_COLOR)
    image_hsv = cv2.cvtColor(image_ip, cv2.COLOR_BGR2HSV)
    cv2.putText(image_ip, "Original image", (50, 50), FONT_ITALIC, 1, (0, 0, 0), 2)
    cv2.imshow('original', image_ip)  # Display the image

    height, width, _ = image_ip.shape
    image_canvas_ip = np.zeros((height, width), np.uint8)
    mask_ip = None  # preventing used before assigned
    rgb_letter = ""

    i = 0
    while True:
        if i == 4:
            break

        if i == 3:  # black color
            lower = np.array([0, 0, 0])
            upper = np.array([50, 50, 50])
            mask_ip = cv2.inRange(image_hsv, lower, upper)

        if i == 0:  # green color
            lower = np.array([35, 150, 20])
            upper = np.array([70, 255, 255])
            mask_ip = cv2.inRange(image_hsv, lower, upper)

        if i == 1:  # blue color
            lower = np.array([70, 150, 20])
            upper = np.array([130, 255, 255])
            mask_ip = cv2.inRange(image_hsv, lower, upper)

        if i == 2:  # red color
            # lower mask (0-10)
            lower_red = np.array([0, 50, 50])
            upper_red = np.array([10, 255, 255])
            mask0 = cv2.inRange(image_hsv, lower_red, upper_red)

            # upper mask (170-180)
            lower_red = np.array([170, 50, 50])
            upper_red = np.array([180, 255, 255])
            mask1 = cv2.inRange(image_hsv, lower_red, upper_red)

            # join my masks
            mask_ip = mask0 + mask1

        # detect contours
        contours, _ = cv2.findContours(mask_ip, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        maskr = np.zeros(mask_ip.shape, np.uint8)
        maskr2 = np.zeros(mask_ip.shape, np.uint8)
        cv2.drawContours(maskr, contours, -1, 255, 1)

        # Preciso de uma imagem de trabalho onde possa preencher e achar o centro para depois escrever na
        # mascara inicial
        cv2.drawContours(maskr2, contours, -1, 255, -1)
        kernel = np.ones((3, 3), 'uint8')

        image_1 = cv2.erode(maskr2, kernel, iterations=3)

        # write letter colors on contours
        contours_m, _ = cv2.findContours(image_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contours_mi in contours_m:
            moments = cv2.moments(contours_mi)
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                if i == 0:
                    rgb_letter = "G"
                elif i == 1:
                    rgb_letter = "B"
                elif i == 2:
                    rgb_letter = "R"
                elif i == 3:
                    rgb_letter = "D"
                cv2.putText(image_canvas_ip, rgb_letter, (cx + 2, cy + 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        image_canvas_ip = cv2.add(image_canvas_ip, maskr)
        i += 1

    image_canvas_ip = cv2.bitwise_not(image_canvas_ip)

    return image_canvas_ip


def main():
    """
    INITIALIZE -----------------------------------------
    """
    # program flags
    background_white = True  # background color flag
    pointer_on = False  # pointer method incomplete
    rect_drawing = False  # rectangle drawing flag
    rect_drawing_mouse = False  # rect draw with the mouse
    circle_drawing = False  # circle drawing flag
    circle_drawing_mouse = False  # circle draw with the mouse
    shake_prevention = False  # shake prevention flag
    image_load_flag = False  # image load flag
    image_prepare_flag = False  # image preparation flag

    # variables
    global background, image_load, image_canvas
    dot_x, dot_y = 0, 0  # pen points
    prev_x, prev_y = 0, 0  # point for continuous draw
    rect_pt1_x, rect_pt1_y, rect_pt2_x, rect_pt2_y = 0, 0, 0, 0  # rectangle drawing points
    circle_pt1_x, circle_pt1_y, circle_pt2_x, circle_pt2_y = 0, 0, 0, 0  # circle drawing points
    # pen variables
    pen_color = (51, 51, 51)
    pen_thickness = 5

    image_copy = None  # preventing used before assigned
    Image_with_Color_Key = None  # preventing used before assigned
    name_of_BLOB_img = None  # preventing used before assigned

    # parse the json file with BGR limits (from color_segmenter.py)
    parser = argparse.ArgumentParser(description="Load a json file with RGB limits and an image to paint on")
    parser.add_argument("-j", "--json", type=str, required=True, help="Full path to json file")
    parser.add_argument("-usp", "--use_shake_prevention", action="store_true", help="Activating shake prevention")
    parser.add_argument("-im", "--image_load", type=str, help="Full path to BLOBX_0 (X ⊂ [3, 6])")
    parser.add_argument("-ip", "--image_prepare", type=str, help="Full path to png file, to clean")
    args = vars(parser.parse_args())

    # activate shake prevention
    if args["use_shake_prevention"]:
        shake_prevention = True

    if args["image_load"]:
        image_load_flag = True

    if args["image_prepare"]:
        image_prepare_flag = True

    # read the json file
    with open(args["json"], "r") as file_handle:
        data = json.load(file_handle)

    # get image load on -im
    if image_load_flag and not image_prepare_flag:
        name_of_BLOB_img = Images_names[args['image_load']]

        Image_to_paint_name = str(name_of_BLOB_img) + '.png'
        Image_with_Color_Key_name = str(name_of_BLOB_img) + '11.png'

        image_load = cv2.imread(Image_to_paint_name, cv2.IMREAD_COLOR)  # read the image from parse
        Image_with_Color_Key = cv2.imread(Image_with_Color_Key_name, cv2.IMREAD_COLOR)  # read the image from parse

    # get image load on -ip
    if image_prepare_flag and not image_load_flag:
        try:
            image_load = prepare_image(args['image_prepare'])
            image_load = cv2.cvtColor(image_load, cv2.IMREAD_COLOR)
            cv2.namedWindow("image load")  # create window for the image
        except ValueError:
            print("Error loading the file, please try again.")

    # print how to use on terminal ******************************************************************

    print(Back.YELLOW + Fore.BLACK + Style.BRIGHT + "\n====================Augmented Reality "
                                                    "Paint====================\n")
    print("\n")
    print("=> Draw and paint using a detected moving object from a chosen camera. You can use the mouse too!")
    print("=> Select an object to be your brush by loading a json file with the respective RGB limits")
    print("=> Run -h command for help" + Style.RESET_ALL)
    print("\n")
    print(Back.CYAN + Fore.BLACK + "=> Interactive Keys:" + Style.BRIGHT + Style.RESET_ALL)
    print("\n")
    print(Fore.YELLOW + "w" + Style.RESET_ALL + " - Save image as png file")
    print(Fore.YELLOW + "ESC or q" + Style.RESET_ALL + " - Quit the program")
    print(Fore.YELLOW + "r" + Style.RESET_ALL + " - Sets the drawing pencil color to RED")
    print(Fore.YELLOW + "g" + Style.RESET_ALL + " - Sets the drawing pencil color to GREEN")
    print(Fore.YELLOW + "b" + Style.RESET_ALL + " - Sets the drawing pencil color to BLUE")
    print(Fore.YELLOW + "m" + Style.RESET_ALL + " - Sets the drawing pencil color to BLACK")
    print(Fore.YELLOW + "+" + Style.RESET_ALL + " - Increases the drawing pencil thickness")
    print(Fore.YELLOW + "-" + Style.RESET_ALL + " - Decreases the drawing pencil thickness")
    print(Fore.YELLOW + "c" + Style.RESET_ALL + " - Clears the drawing screen")
    print(Fore.YELLOW + "a" + Style.RESET_ALL + " - Allows the user to clean the screen with the pointer, "
                                                "like an eraser")
    print(Fore.YELLOW + "f" + Style.RESET_ALL + " - Allows the user to switch between a black and white backgrounds")
    print(Fore.YELLOW + "p" + Style.RESET_ALL + " - Enables the pointer mode i.e allows the user to move a pointer on "
                                                "the screen")
    print(Fore.YELLOW + "j" + Style.RESET_ALL + " - Draws a rectangle using mouse events")
    print(Fore.YELLOW + "o" + Style.RESET_ALL + " - Draws a circle using mouse events")
    print(Fore.YELLOW + "l" + Style.RESET_ALL + " - The use of the following instructions ('s', 'e', 'j' and 'o') "
                                                "requires the pressing of the letter 'l' to draw the object with "
                                                "the desired dimensions")
    print(Fore.YELLOW + "s" + Style.RESET_ALL + " - Draws a rectangle")
    print(Fore.YELLOW + "e" + Style.RESET_ALL + " - Draws a circle")
    print("\n")

    # print json file then close
    print("Loaded RGB limits: \n")
    print(data)
    print("\n")
    # file_handle.close()

    # ***************************************

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

    # create canvas image to join draw and video
    image_canvas = np.zeros((486, 864, 3), np.uint8)

    """
    EXECUTION -----------------------------------------
    """
    while True:
        # read the image
        _, image = capture.read()
        image = cv2.resize(image, (864, 486))  # resize the capture window
        image = cv2.flip(image, 1)  # flip video capture

        # transform the image
        mask_im = cv2.inRange(image, mins_pcss, maxs_pcss)  # colors mask
        image_segmenter = cv2.bitwise_and(image, image, mask=mask_im)

        # get contours
        contours, hierarchy = cv2.findContours(mask_im.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # create the rectangle over object and draw on the background
        for contours in contours:
            (x, y, w, h) = cv2.boundingRect(contours)

            # draw the rectangle only if the area is > 3000 px for preventing multiple detections
            if cv2.contourArea(contours) < 3000:
                continue

            # draw the rectangle over detected object
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # center of the rectangle
            dot_x = x + w / 2
            dot_y = y + h / 2

            # Draw red cross under the centroid of the detected object
            cv2.putText(image, '+', (int(dot_x), int(dot_y)), FONT_ITALIC, 1, (255, 0, 0), 2, LINE_8)

            # mode for cursor only
            if not pointer_on:
                # draw in the background
                if prev_x == 0 and prev_y == 0:  # skip first iteration
                    prev_x, prev_y = dot_x, dot_y

                # Activating shake prevention
                if shake_prevention:
                    if abs(prev_x - dot_x) < 50 and abs(prev_y - dot_y) < 50:
                        cv2.line(background, (int(prev_x), int(prev_y)), (int(dot_x),
                                                                          int(dot_y)), pen_color, pen_thickness)
                        cv2.line(image_canvas, (int(prev_x), int(prev_y)), (int(dot_x), int(dot_y)), pen_color,
                                 pen_thickness)

                        # painting on loaded image
                        if image_load_flag or image_prepare_flag:

                            # draw lines on the load image
                            if abs(prev_x - dot_x) < 50 and abs(prev_y - dot_y) < 50:
                                cv2.line(image_load, (int(prev_x), int(prev_y)), (int(dot_x),
                                                                                  int(dot_y)), pen_color, pen_thickness)

                        prev_x, prev_y = dot_x, dot_y  # reset coordinates
                    else:
                        prev_x, prev_y = 0, 0
                else:
                    cv2.line(background, (int(prev_x), int(prev_y)), (int(dot_x),
                                                                      int(dot_y)), pen_color, pen_thickness)
                    cv2.line(image_canvas, (int(prev_x), int(prev_y)), (int(dot_x), int(dot_y)), pen_color,
                             pen_thickness)
                    prev_x, prev_y = dot_x, dot_y

            # point only mode--------------------------------------------------
            else:
                background = cv2.imread("temp.png")
                cv2.putText(background, '+', (int(dot_x), int(dot_y)), FONT_ITALIC, 1, (255, 0, 0), 2, LINE_8)

        # show the concatenated window
        if not image_load_flag and not image_prepare_flag:
            # deepcopy the original image (creates a full new copy without references)
            image_copy = copy.deepcopy(image)

            # put text on images
            cv2.putText(image, "Video Capture", (50, 50), FONT_ITALIC, 1, (0, 0, 0), 2)
            if background_white:
                cv2.putText(background, "Drawing Area", (50, 50), FONT_ITALIC, 1, (0, 0, 0), 2)
            else:
                cv2.putText(background, "Drawing Area", (50, 50), FONT_ITALIC, 1, (255, 255, 255), 2)

            cv2.putText(image_copy, "Video plus Drawing", (50, 50), FONT_ITALIC, 1, (0, 0, 0), 2)
            cv2.putText(image_segmenter, "Object Mask", (50, 50), FONT_ITALIC, 1, (255, 255, 255), 2)

            # put instructions on background
            text_pos_width = 690
            text_pos_height = 50
            text_space = 20
            text_scale = 0.4
            # if background flip
            if background_white:
                text_color = (0, 0, 0)
            else:
                text_color = (255, 255, 255)

            cv2.putText(background, "w - Save image", (text_pos_width, text_pos_height),
                        cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, 1)
            cv2.putText(background, "r - Sets color to RED", (text_pos_width, text_pos_height + text_space),
                        cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, 1)
            cv2.putText(background, "g - Sets color to GREEN", (text_pos_width, text_pos_height + text_space * 2),
                        cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, 1)
            cv2.putText(background, "b - Sets color to BLUE", (text_pos_width, text_pos_height + text_space * 3),
                        cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, 1)
            cv2.putText(background, "m - Sets color to BLACK", (text_pos_width, text_pos_height + text_space * 4),
                        cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, 1)
            cv2.putText(background, "+ - Increases thickness", (text_pos_width, text_pos_height + text_space * 5),
                        cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, 1)
            cv2.putText(background, "- - Decreases thickness", (text_pos_width, text_pos_height + text_space * 6),
                        cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, 1)
            cv2.putText(background, "c - Clear", (text_pos_width, text_pos_height + text_space * 7),
                        cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, 1)
            cv2.putText(background, "a - Eraser", (text_pos_width, text_pos_height + text_space * 8),
                        cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, 1)
            cv2.putText(background, "f - Flip backgrounds", (text_pos_width, text_pos_height + text_space * 9),
                        cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, 1)
            cv2.putText(background, "p - Pointer", (text_pos_width, text_pos_height + text_space * 10),
                        cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, 1)
            cv2.putText(background, "j - Mouse rectangle", (text_pos_width, text_pos_height + text_space * 11),
                        cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, 1)
            cv2.putText(background, "o - Mouse circle", (text_pos_width, text_pos_height + text_space * 12),
                        cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, 1)
            cv2.putText(background, "s - Rectangle", (text_pos_width, text_pos_height + text_space * 13),
                        cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, 1)
            cv2.putText(background, "e - Circle", (text_pos_width, text_pos_height + text_space * 14),
                        cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, 1)
            cv2.putText(background, "l - Lock shape", (text_pos_width, text_pos_height + text_space * 15),
                        cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, 1)

            # merge the video and the drawing
            image_gray = cv2.cvtColor(image_canvas, cv2.COLOR_BGR2GRAY)
            _, image_inverse = cv2.threshold(image_gray, 50, 255, cv2.THRESH_BINARY_INV)

            # join video and drawing
            image_inverse = cv2.cvtColor(image_inverse, cv2.COLOR_GRAY2BGR)
            image_copy = cv2.bitwise_and(image_copy, image_inverse)
            image_copy = cv2.bitwise_or(image_copy, image_canvas)

            # horizontally concatenating the two frames
            final_frame_h1 = cv2.hconcat((image, background))
            final_frame_h2 = cv2.hconcat((image_segmenter, image_copy))

            # join frames
            final_frame = cv2.vconcat((final_frame_h1, final_frame_h2))

            # Show the concatenated frame using image show
            cv2.imshow('frame', final_frame)

        # show the loaded image and the video only
        if image_load_flag and not image_prepare_flag:
            # make the image load the same size as the camera image so that it can be painted in all extension
            image_load = cv2.resize(image_load, (864, 486))  # resize the capture window
            Image_with_Color_Key = cv2.resize(Image_with_Color_Key, (864, 486))  # resize the capture window

            if background_white:
                cv2.putText(background, "Drawing Area", (50, 50), FONT_ITALIC, 1, (0, 0, 0), 2)
            else:
                cv2.putText(background, "Drawing Area", (50, 50), FONT_ITALIC, 1, (255, 255, 255), 2)

            # show image
            cv2.putText(image, "Video Capture", (50, 50), FONT_ITALIC, 1, (0, 0, 0), 2)
            cv2.putText(image_load, "Image to paint", (50, 50), FONT_ITALIC, 1, (0, 0, 0), 2)
            cv2.putText(Image_with_Color_Key, "Color key", (50, 50), FONT_ITALIC, 1, (0, 0, 0), 2)

            # put instructions on background
            text_pos_width = 690
            text_pos_height = 50
            text_space = 20
            text_scale = 0.4
            # if background flip
            if background_white:
                text_color = (0, 0, 0)
            else:
                text_color = (255, 255, 255)

            cv2.putText(background, "w - Save image", (text_pos_width, text_pos_height),
                        cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, 1)
            cv2.putText(background, "r - Sets color to RED", (text_pos_width, text_pos_height + text_space),
                        cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, 1)
            cv2.putText(background, "g - Sets color to GREEN", (text_pos_width, text_pos_height + text_space * 2),
                        cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, 1)
            cv2.putText(background, "b - Sets color to BLUE", (text_pos_width, text_pos_height + text_space * 3),
                        cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, 1)
            cv2.putText(background, "m - Sets color to BLACK", (text_pos_width, text_pos_height + text_space * 4),
                        cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, 1)
            cv2.putText(background, "+ - Increases thickness", (text_pos_width, text_pos_height + text_space * 5),
                        cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, 1)
            cv2.putText(background, "- - Decreases thickness", (text_pos_width, text_pos_height + text_space * 6),
                        cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, 1)
            cv2.putText(background, "c - Clear", (text_pos_width, text_pos_height + text_space * 7),
                        cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, 1)
            cv2.putText(background, "a - Eraser", (text_pos_width, text_pos_height + text_space * 8),
                        cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, 1)
            cv2.putText(background, "f - Flip backgrounds", (text_pos_width, text_pos_height + text_space * 9),
                        cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, 1)
            cv2.putText(background, "p - Pointer", (text_pos_width, text_pos_height + text_space * 10),
                        cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, 1)
            cv2.putText(background, "j - Mouse rectangle", (text_pos_width, text_pos_height + text_space * 11),
                        cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, 1)
            cv2.putText(background, "o - Mouse circle", (text_pos_width, text_pos_height + text_space * 12),
                        cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, 1)
            cv2.putText(background, "s - Rectangle", (text_pos_width, text_pos_height + text_space * 13),
                        cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, 1)
            cv2.putText(background, "e - Circle", (text_pos_width, text_pos_height + text_space * 14),
                        cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, 1)
            cv2.putText(background, "l - Lock shape", (text_pos_width, text_pos_height + text_space * 15),
                        cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, 1)

            # concatenate
            paint_frame_v1 = cv2.vconcat((Image_with_Color_Key, image_load))
            paint_frame_v2 = cv2.vconcat((image, background))
            paint_frame_h = cv2.hconcat((paint_frame_v2, paint_frame_v1))

            # Show the concatenated frame using image show.
            cv2.imshow('paint_frame', paint_frame_h)

        # image prepare flag ********************************************************************
        if not image_load_flag and image_prepare_flag:
            # draw on load image with mouse
            draw_on_image = partial(mouse_draw, pen_color=pen_color, pen_thickness=pen_thickness)
            cv2.putText(image_load, "Image to paint", (50, 50), FONT_ITALIC, 1, (0, 0, 0), 2)
            cv2.setMouseCallback('image load', draw_on_image)  # draw with mouse
            cv2.imshow("image load", image_load)

            cv2.namedWindow("Video Capture")
            cv2.putText(image, "Video Capture", (50, 50), FONT_ITALIC, 1, (0, 0, 0), 2)
            cv2.imshow("Video Capture", image)

        """
        interactive keys (k) -----------------------------------------
        """
        # ESC and q to close
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or k == ord("q"):
            break

        # clean the screen
        if k == ord("c"):
            if background_white:
                background.fill(255)
                image_canvas.fill(0)
                cv2.imwrite('./temp' + '.png', background)  # Save the drawing for temp use
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
        if k == ord("m"):
            if background_white:
                pen_color = (51, 51, 51)
                print("YOU SELECT BLACK COLOR")
            else:
                pen_color = (255, 255, 255)
                print("YOU SELECT WHITE COLOR")

        # thickness
        if k == 43:  # character + to increase thickness
            pen_thickness += 1
            print("THICKNESS: " + str(pen_thickness))

        if k == 45:  # character - to decrease thickness
            pen_thickness -= 1
            if pen_thickness <= 0:
                pen_thickness = 1
            print("THICKNESS: " + str(pen_thickness))

        # erase
        if k == ord("a"):
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
                pen_color = (51, 51, 51)

        # pointer mode
        if k == ord("p"):
            if pointer_on:
                pointer_on = False
                background = cv2.imread("temp.png")
            else:
                pointer_on = True
                cv2.imwrite('./temp' + '.png', background)  # Save the drawing for temp use
                background = cv2.imread("temp.png")

        # save the draw in png file
        if k == ord("w"):
            if not image_load_flag and not image_prepare_flag:
                cv2.imwrite('./drawing_' + str(ctime()) + '.png', background)  # Save the drawing
                cv2.imwrite('./drawing_cam_' + str(ctime()) + '.png', image_copy)  # Save the drawing on the camera

            if image_load_flag:
                Image_painted = str(name_of_BLOB_img) + '4.png'
                cv2.imwrite(Image_painted, image_load)  # Save the drawing painted by AR
                paint_evaluation(name_of_BLOB_img)
                break

            if image_prepare_flag:
                cv2.imwrite('./drawing_cam_' + str(ctime()) + '.png', image_load)  # Save the drawing on the camera
            print("DRAWINGS SAVED AS A .PNG FILE")

        # draw a rectangle with mouse events
        if k == ord("j"):
            cv2.imwrite('./temp' + '.png', background)  # Save the drawing for temp use
            background = cv2.imread("temp.png")
            cv2.namedWindow('Image_Window')
            rect_drawing_mouse = True

        if rect_drawing_mouse:
            rectangle = partial(shape, mode=str('rectangle'), pen_color=pen_color, pen_thickness=pen_thickness)
            cv2.setMouseCallback('Image_Window', rectangle)
            cv2.imshow('Image_Window', background)

        if k == ord("l") and rect_drawing_mouse:
            rect_drawing_mouse = False
            cv2.destroyWindow("Image_Window")

        # draw a circle with mouse events
        if k == ord("o"):
            cv2.imwrite('./temp' + '.png', background)  # Save the drawing for temp use
            background = cv2.imread("temp.png")
            cv2.namedWindow('Image_Window')
            circle_drawing_mouse = True

        if circle_drawing_mouse:
            circle = partial(shape, mode=str('circle'), pen_color=pen_color, pen_thickness=pen_thickness)
            cv2.setMouseCallback('Image_Window', circle)
            cv2.imshow('Image_Window', background)

        if k == ord("l") and circle_drawing_mouse:
            circle_drawing_mouse = False
            cv2.destroyWindow("Image_Window")

        # draw a rectangle----------------------------------------------------------------------
        if k == ord("s"):
            cv2.imwrite('./temp' + '.png', background)  # Save the drawing for temp use
            background = cv2.imread("temp.png")
            rect_drawing = True
            rect_pt1_x = int(dot_x)
            rect_pt1_y = int(dot_y)

        if rect_drawing:
            background = cv2.imread("temp.png")
            rect_pt2_x = int(dot_x)
            rect_pt2_y = int(dot_y)
            cv2.rectangle(background, (rect_pt1_x, rect_pt1_y), (rect_pt2_x, rect_pt2_y), pen_color, pen_thickness)
            a = rect_pt2_x
            b = rect_pt2_y
            if a != rect_pt2_x | b != rect_pt2_y:
                cv2.rectangle(background, (rect_pt1_x, rect_pt1_y), (rect_pt2_x, rect_pt2_y), pen_color, pen_thickness)

        if k == ord("l") and rect_drawing:
            rect_pt2_x = int(dot_x)
            rect_pt2_y = int(dot_y)
            rect_drawing = False
            cv2.rectangle(background, (rect_pt1_x, rect_pt1_y), (rect_pt2_x, rect_pt2_y), pen_color, pen_thickness)
            cv2.rectangle(image_canvas, (rect_pt1_x, rect_pt1_y), (rect_pt2_x, rect_pt2_y), pen_color, pen_thickness)

        # draw a circle------------------------------------------------------------------------
        if k == ord("e"):
            cv2.imwrite('./temp' + '.png', background)  # Save the drawing for temp use
            background = cv2.imread("temp.png")
            circle_drawing = True
            circle_pt1_x = int(dot_x)
            circle_pt1_y = int(dot_y)

        if circle_drawing:
            background = cv2.imread("temp.png")
            circle_pt2_x = int(dot_x)
            circle_pt2_y = int(dot_y)

            try:
                radius = math.pow(((math.pow(circle_pt1_x, 2) - math.pow(circle_pt2_x, 2)) + (
                        math.pow(circle_pt1_y, 2) - math.pow(circle_pt2_y, 2))), 1 / 2)
                cv2.circle(background, (circle_pt2_x, circle_pt2_y), int(radius), pen_color, pen_thickness)
                a = circle_pt1_x
                b = circle_pt1_y
                if a != circle_pt1_x | b != circle_pt1_y:
                    radius = math.pow(((math.pow(circle_pt1_x, 2) - math.pow(circle_pt2_x, 2)) + (
                            math.pow(circle_pt1_y, 2) - math.pow(circle_pt2_y, 2))), 1 / 2)
                    cv2.circle(background, (circle_pt2_x, circle_pt2_y), int(radius), pen_color, pen_thickness)
            except ValueError:
                pass

        if k == ord("l") and circle_drawing:
            circle_pt2_x = int(dot_x)
            circle_pt2_y = int(dot_y)
            circle_drawing = False

            try:
                radius = math.pow(((math.pow(circle_pt1_x, 2) - math.pow(circle_pt2_x, 2)) + (
                        math.pow(circle_pt1_y, 2) - math.pow(circle_pt2_y, 2))), 1 / 2)
                cv2.circle(background, (circle_pt2_x, circle_pt2_y), int(radius), pen_color, pen_thickness)
                cv2.circle(image_canvas, (circle_pt2_x, circle_pt2_y), int(radius), pen_color, pen_thickness)
                a = circle_pt1_x
                b = circle_pt1_y
                if a != circle_pt1_x | b != circle_pt1_y:
                    radius = math.pow(((math.pow(circle_pt1_x, 2) - math.pow(circle_pt2_x, 2)) + (
                            math.pow(circle_pt1_y, 2) - math.pow(circle_pt2_y, 2))), 1 / 2)
                    cv2.circle(background, (circle_pt2_x, circle_pt2_y), int(radius), pen_color, pen_thickness)
                    cv2.circle(image_canvas, (circle_pt2_x, circle_pt2_y), int(radius), pen_color, pen_thickness)
            except ValueError:
                pass

    """
    FINALIZATION -----------------------------------------
    """
    capture.release()  # free the webcam for other use
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
