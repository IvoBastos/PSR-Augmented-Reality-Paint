#!/usr/bin/python3


import argparse
import json
import math
from functools import partial
from time import time, ctime, sleep
import cv2
import numpy as np
from cv2 import FONT_ITALIC, LINE_8
from matplotlib import pyplot as plt

# dictionary with ranges
ranges_pcss = {"b": {"min": 100, "max": 256},
               "g": {"min": 100, "max": 256},
               "r": {"min": 100, "max": 256},
               }

drawing = False # true if mouse is pressed
# mode = str('rectangle') # if 'rectangle', draw rectangle.
ix,iy = -1,-1

# create a white image background dim 600*400
background = np.zeros((422, 750, 3), np.uint8)
background.fill(255)


def shape(event,x,y,flags,params,mode):
    global ix,iy,drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        #value of variable draw will be set to True, when you press DOWN left mouse button
        drawing = True
        #mouse location is captured here
        ix,iy=x,y

    elif event == cv2.EVENT_MOUSEMOVE:
      #Dragging the mouse at this junture
      if drawing == True:
        if mode == 'rectangle':
            #If draw is True then it means you've cliked on the left mouse button
            #Here we will draw a rectangle from previos position to the x,y where the mouse is currently located
            cv2.rectangle(background, (ix, iy), (x, y), (0, 255, 0), 3)
            # Nas ultimas cordenadas devo desenhar um retangulo da cor do fundo
            a = x
            b = y
            if a != x | b != y:
                cv2.rectangle(background, (ix, iy), (x, y), (255, 255, 255), -1)
        if mode == 'circle':
            radius=math.pow(((math.pow(x,2)-math.pow(ix,2))+(math.pow(y,2)-math.pow(iy,2))),1/2)
            cv2.circle(background,(ix, iy), int(radius), (0, 0, 255),3)
            a = x
            b = y
            if a != x | b != y:
                radius = math.pow(((math.pow(x, 2) - math.pow(ix, 2)) + (math.pow(y, 2) - math.pow(iy, 2))),1/2)
                cv2.circle(background,(ix, iy), int(radius), (255, 255, 255), -1)

    elif event == cv2.EVENT_LBUTTONUP:
     drawing = False
     if mode == 'rectangle':
        #As soon as you release the mouse button, variable draw will be set as False
        #Here we are completing to draw the rectangle on image window
        # background=clone
        cv2.rectangle(background, (ix, iy), (x, y), (0, 255, 0), 2)

     if mode == 'circle':
         radius = math.pow(((math.pow(x, 2) - math.pow(ix, 2)) + (math.pow(y, 2) - math.pow(iy, 2))), 1 / 2)
         cv2.circle(background, (ix, iy), int(radius), (0, 0, 255), 3)



#COMIT ANALTINO
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
    # cv2.namedWindow(window_segmented, cv2.WINDOW_AUTOSIZE)

    # create a white image background dim 600*400
    # background = np.zeros((422, 750, 3), np.uint8)
    # background.fill(255)

    # window for display background/draw area
    draw_area = "Draw Area"
    # cv2.namedWindow(draw_area)

    # merged camera and drawing
    #merged_area = "Interactive Drawing"
    # cv2.namedWindow(merged_area)
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
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # dot in the center of the rectangle
            dot_x = x + w / 2
            dot_y = y + h / 2
            # cv2.circle(image, (int(dot_x), int(dot_y)), 10, (0, 0, 0), cv2.FILLED)
            #Draw cruz vermelha sobre o centroid da imagem
            cv2.putText(image, '+', (int(dot_x), int(dot_y)), FONT_ITALIC,1 ,(255,0,0),2,LINE_8)

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

        # merge the video and the drawing ----------------------------INCOMPLETE DOESN'T DRAW THE BLACK COLOR
        #image_merged = cv2.addWeighted(image, 0.5, background, 0.5, 0) # merged the images to draw on the video

        image_gray = cv2.cvtColor(image_canvas, cv2.COLOR_BGR2GRAY)
        _, image_inverse = cv2.threshold(image_gray, 50, 255, cv2.THRESH_BINARY_INV)

        final_frame_h1 = cv2.hconcat((image, background))

        image_inverse = cv2.cvtColor(image_inverse, cv2.COLOR_GRAY2BGR)
        image = cv2.bitwise_and(image, image_inverse)
        image = cv2.bitwise_or(image, image_canvas)
        # cv2.imshow(merged_area, image)


        # horizintally concatenating the two frames.

        final_frame_h2 = cv2.hconcat((image_segmenter, image))
        final_frame = cv2.vconcat((final_frame_h1,final_frame_h2))

        # Show the concatenated frame using imshow.
        cv2.imshow('frame',final_frame)

        """
        interactive keys (k) -----------------------------------------
        """
        # ESC to close
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or k == ord("q"):
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
        if k == 114: #ord("r"):
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
        if k == 43: #caracter + "Aumentar linha"
            pen_thickness += 1
            print("THICKNESS: " + str(pen_thickness))

        if k == 45: #caracter - "Diminuir linha"
            pen_thickness -= 1
            if pen_thickness < 0:
                pen_thickness = 0
            print("THICKNESS: " + str(pen_thickness))

        # # erase
        # if k == ord("e"):
        #     if background_white:
        #         pen_color = (255, 255, 255)
        #         print("YOU SELECT ERASER")
        #     else:
        #         pen_color = (0, 0, 0)
        #         print("YOU SELECT ERASER")

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
        if k == ord("s"):

            cv2.namedWindow('Image_Window')
            rectangle = partial(shape, mode=str('rectangle'))
            cv2.setMouseCallback('Image_Window',rectangle)
            cv2.imshow('Image_Window',background)

        # draw a circle------------------------------------------ INCOMPLETE, DOESN'T WORK PROPERLY
        if k == ord("e"):

            cv2.namedWindow('Image_Window')
            circle=partial(shape,mode=str('circle'))
            cv2.setMouseCallback('Image_Window',circle)
            cv2.imshow('Image_Window',background)

        if k == ord("L") and circle_drawing:
            circle_pt2_x = int(dot_x)
            circle_pt2_y = int(dot_y)
            circle_drawing = False
            cv2.ellipse(background, (circle_pt1_x, circle_pt1_y),
                        (circle_pt2_x, circle_pt2_y), 0, 0, 360, pen_color, cv2.FILLED)
        # erase
        if k == ord("w"):
            cv2.imwrite('./drawing_' + str(ctime()) + '.png', background)
    """
    FINALIZATION -----------------------------------------
    """
    capture.release()  # free the webcam for other use
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
