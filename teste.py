#!/usr/bin/python3
import numpy as np
import cv2
import matplotlib.pyplot as plt
#COMIT ANALTINO

def main():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    #bg = [[[0] * len(frame[0]) for _ in xrange(len(frame))] for _ in xrange(3)]

    while(True):
        ret, frame = cap.read()
        # Resizing down the image to fit in the screen.
        #frame = cv2.resize(frame, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_CUBIC)

        # creating another frame.
        #channels = cv2.split(frame)
        #frame_merge = cv2.merge(channels)

        # horizintally concatenating the two frames.
        final_frame = cv2.hconcat((frame, frame,frame))

        # Show the concatenated frame using imshow.
        cv2.imshow('frame',final_frame)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
