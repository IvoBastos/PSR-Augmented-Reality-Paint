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

# dictionary with ranges
ranges_pcss = {"b": {"min": 100, "max": 256},
               "g": {"min": 100, "max": 256},
               "r": {"min": 100, "max": 256},
               }

# global variables
drawing = False  # true in drawing shame mode
ix, iy, Drag = -1, -1, False  # used in def shape
pt1_y, pt1_x, moving_mouse = 0, 0, False  # used in def mouse_draw

# create a white image background
background = np.zeros((422, 750, 3), np.uint8)
background.fill(255)

# image load on parse -im
image_load = None


def shape(event, x, y, flags, params, mode, pen_color, pen_thickness):
    global ix, iy, Drag, background

    if event == cv2.EVENT_LBUTTONDOWN and not Drag:
        # value of variable draw will be set to True, when you press DOWN left mouse button
        # mouse location is captured here
        ix, iy = x, y
        Drag = True
        # cv2.rectangle(background, (ix, iy), (x, y), pen_color, pen_thickness)

    elif event == cv2.EVENT_MOUSEMOVE:
        # Dragging the mouse at this juncture

        if Drag:
            # if background_white:
            #     background.fill(255)
            # else:
            #     background.fill(0)
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
                        radius = math.pow(((math.pow(x, 2) - math.pow(ix, 2)) + (math.pow(y, 2) - math.pow(iy, 2))), 1 / 2)
                        cv2.circle(background, (ix, iy), int(radius), pen_color, pen_thickness)
                except:
                    pass

    elif event == cv2.EVENT_LBUTTONDOWN and Drag:
        Drag = False

        if mode == 'rectangle' and not Drag:
            # As soon as you release the mouse button, variable draw will be set as False
            # Here we are completing to draw the rectangle on image window
            cv2.rectangle(background, (ix, iy), (x, y), pen_color, pen_thickness)

        if mode == 'circle':
            radius = math.pow(((math.pow(x, 2) - math.pow(ix, 2)) + (math.pow(y, 2) - math.pow(iy, 2))), 1 / 2)
            cv2.circle(background, (ix, iy), int(radius), pen_color, pen_thickness)

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


def prepare_image(imagemTratar):

    image = cv2.imread(imagemTratar, cv2.IMREAD_COLOR)
    imagehsv= cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imshow('original', image)  # Display the image

    height, width, _ = image.shape
    image_canvas = np.zeros((height, width), np.uint8)

    i = 0
    while True:
        if i == 4:
            break

        if i == 3: #apanha o preto +/-
            lower = np.array([0,0,0])
            upper = np.array([50,50,50])
            mask = cv2.inRange(imagehsv, lower, upper)

        if i == 0: #apanha o verde +/-
            lower = np.array([35,150,20])
            upper = np.array([70,255,255])
            mask = cv2.inRange(imagehsv, lower, upper)

        if i==1: #apanha o azul +/-
            lower = np.array([70, 150, 20])
            upper = np.array([130, 255, 255])
            mask = cv2.inRange(imagehsv, lower, upper)

        if i == 2: #apanha o vermelho +/-
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

        # deteta os contornos da cor a trabalhar
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        maskr = np.zeros(mask.shape, np.uint8)
        maskr2 = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(maskr, contours, -1, (255), 1)

        # Preciso de uma imagem de trabalho onde possa preencher achar e achar o centro para depois escrever na mascara inicial
        cv2.drawContours(maskr2, contours, -1, (255), -1)
        kernel = np.ones((3, 3), 'uint8')

        image_1 = cv2.erode(maskr2, kernel, iterations=3)

        # Escrever o numero da cor no centro do contorno a pintar
        contoursm, _ = cv2.findContours(image_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contoursmi in contoursm:
            M = cv2.moments(contoursmi)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.putText(image_canvas, str(i+1), (cx+2, cy+2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        image_canvas = cv2.add(image_canvas, maskr)
        i = i + 1

    image_canvas = cv2.bitwise_not(image_canvas)
    #cv2.imshow('pinta', image_canvas)  # Display the image

    return image_canvas


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
    image_prepare = False  # image preparation flag

    # variables
    global background, image_load
    dot_x, dot_y = 0, 0  # pen points
    prev_x, prev_y = 0, 0  # point for continuous draw
    rect_pt1_x, rect_pt1_y, rect_pt2_x, rect_pt2_y = 0, 0, 0, 0  # rectangle drawing points
    circle_pt1_x, circle_pt1_y, circle_pt2_x, circle_pt2_y = 0, 0, 0, 0  # circle drawing points
    # pen variables
    pen_color = (51, 51, 51)
    pen_thickness = 5

    # image copy of the camara to draw on
    image_copy = None  # preventing used before assignment bug

    # parse the json file with BGR limits (from color_segmenter.py)
    parser = argparse.ArgumentParser(description="Load a json file with RGB limits")
    parser.add_argument("-j", "--json", type=str, required=True, help="Full path to json file")
    parser.add_argument("-usp", "--use_shake_prevention", action="store_true", help="Activating shake prevention")
    parser.add_argument("-im", "--image_load", type=str, help="Full path to png file")
    parser.add_argument("-ip", "--image_prepare", type=str, help="Full path to png file, to clean")
    args = vars(parser.parse_args())

    # activate shake prevention
    if args["use_shake_prevention"]:
        shake_prevention = True

    if args["image_load"]:
        image_load_flag = True

    if args["image_prepare"]:
        image_prepare = True

    # read the json file
    with open(args["json"], "r") as file_handle:
        data = json.load(file_handle)

    if image_load_flag and not image_prepare:
        cv2.namedWindow("image load")  # create window for the image
        image_load = cv2.imread(args['image_load'], cv2.IMREAD_COLOR)  # read the image from parse

    if image_prepare:
        try:
            image_load_flag = True
            image_load = prepare_image(args['image_prepare'])
            image_load = cv2.cvtColor(image_load, cv2.IMREAD_COLOR)
            # cv2.namedWindow("image load")  # create window for the image
        except:
            print("Ocoreu um erro a carregar o ficheiro para tratamento")

    # print json file then close
    print(data)
    file_handle.close()

    # ***************************************

    # ADD PRINT() WITH List of letters used to interact with the program

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
    image_canvas = np.zeros((422, 750, 3), np.uint8)

    """
    EXECUTION -----------------------------------------
    """
    while True:
        # read the image
        _, image = capture.read()
        image = cv2.resize(image, (750, 422))  # resize the capture window
        image = cv2.flip(image, 1)  # flip video capture

        # transform the image
        mask = cv2.inRange(image, mins_pcss, maxs_pcss)  # colors mask
        image_segmenter = cv2.bitwise_and(image, image, mask=mask)

        # get contours
        contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # mouse callback on load image
        if image_load_flag and not image_prepare:
            # draw on load image with mouse
            draw_on_image = partial(mouse_draw, pen_color=pen_color, pen_thickness=pen_thickness)
            cv2.setMouseCallback('image load', draw_on_image)

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
                        if image_load_flag:

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
                # if background_white:
                #     background.fill(255)
                # else:
                #     background.fill(0)
                # background.fill(255)
                # cv2.circle(background, (int(dot_x), int(dot_y)), pen_thickness, pen_color, cv2.FILLED)
                cv2.putText(background, '+', (int(dot_x), int(dot_y)), FONT_ITALIC, 1, (255, 0, 0), 2, LINE_8)

        # show the concatenated window
        if not image_load_flag:
            # merge the video and the drawing
            image_gray = cv2.cvtColor(image_canvas, cv2.COLOR_BGR2GRAY)
            _, image_inverse = cv2.threshold(image_gray, 50, 255, cv2.THRESH_BINARY_INV)

            # join frames
            final_frame_h1 = cv2.hconcat((image, background))

            # deepcopy the original image (creates a full new copy without references)
            image_copy = copy.deepcopy(image)

            # join video and drawing
            image_inverse = cv2.cvtColor(image_inverse, cv2.COLOR_GRAY2BGR)
            image_copy = cv2.bitwise_and(image_copy, image_inverse)
            image_copy = cv2.bitwise_or(image_copy, image_canvas)

            # horizontally concatenating the two frames.
            final_frame_h2 = cv2.hconcat((image_segmenter, image_copy))
            final_frame = cv2.vconcat((final_frame_h1, final_frame_h2))

            # Show the concatenated frame using imshow.
            cv2.imshow('frame', final_frame)

        # show the loaded image and the video only
        if image_load_flag:
            # make the image load the same size as the camera image so that it can be painted in all extension
            image_load = cv2.resize(image_load, (750, 422))  # resize the capture window

            # show image (THIS WINDOWS CAN'T BE CONCATENATED BECAUSE IMAGE_LOAD USES MOUSE CALLBACK TO PAINT WITH
            # THE MOUSE)
            cv2.namedWindow("Video Capture")
            cv2.imshow("Video Capture", image)
            cv2.imshow("image load", image_load)

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
                # if background_white:
                #     background.fill(255)
                # else:
                #     background.fill(0)
                background = cv2.imread("temp.png")
            else:
                pointer_on = True
                cv2.imwrite('./temp' + '.png', background)  # Save the drawing for temp use
                background = cv2.imread("temp.png")

        # save the draw in png file
        if k == ord("w"):
            if not image_load_flag:
                cv2.imwrite('./drawing_' + str(ctime()) + '.png', background)  # Save the drawing
                cv2.imwrite('./drawing_cam_' + str(ctime()) + '.png', image_copy)  # Save the drawing on the camera

            if image_load_flag:
                cv2.imwrite('./drawing_' + str(ctime()) + '.png', image_load)  # Save the drawing painted by AR

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
        if k == ord("º"):
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
            except:
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
            except:
                pass

    """
    FINALIZATION -----------------------------------------
    """
    capture.release()  # free the webcam for other use
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
