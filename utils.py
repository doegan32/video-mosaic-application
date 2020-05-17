# -*- coding: utf-8 -*-
"""
Created on Tue May  5 12:46:36 2020

@author: donal
"""

import cv2 as cv
import numpy as np
import os

def read_video(video, orientation = 'landscape', display = True):
    """ input: video - address of video to be processed
               orientation - 'portrait' or 'landscape' depending on phone's orientation
               displayOff - boolean to turn off video dispay
        return: numpy array containing frames of video
    """
    
    vid = cv.VideoCapture(video)
    frames = []
    ret = True
    
    while ret:
        ret, im = vid.read()
        
        if ret:
            if (orientation == 'portrait'):
                im = cv.rotate(im, cv.ROTATE_90_CLOCKWISE) # rotate images if phone was rotated
            frames.append(im)
            
            if display:
                cv.namedWindow('input video', cv.WINDOW_NORMAL)
                cv.resizeWindow('input video', int(im.shape[1]/2), int(im.shape[0]/2) )
                cv.moveWindow('input video', 0, 0) 
                cv.imshow("input video", im)
                key = cv.waitKey(27)
                if key == 27:
                    cv.destroyAllWindows()

    vid.release()
    cv.destroyAllWindows()
    frames = np.array(frames)
    
    print("\nnumber of frames: {:d}\nshape of frame: {:d} x {:d}\nnumber of colour channels: {:d}\n"
          .format(frames.shape[0],frames.shape[1],frames.shape[2],frames.shape[3]))

    return frames

def select_keyframes(frames, frameRate = 10, display = False):
    """ input: frames - numpy array of video frames
               n - take every nth frame
        return: numpy array of selected frames """

    m = int(frames.shape[0]/frameRate)
    
    if( m%2 == 0):
        m = m - 1
    
    nFrames = np.zeros((m, frames.shape[1], frames.shape[2], frames.shape[3]), dtype = np.uint8)
    for i in range(m):
        nFrames[i] = frames[frameRate * i]
        
        
    if display:
        for i in range(m):
            display_image(nFrames[i],'frame'+str(i), 500, (int(nFrames[i].shape[0]/2), int(nFrames[i].shape[1]/2)))

    print("\nTaking every {:d}th frame.\nNumber of frames used is {:d}.\n".format(frameRate,m))
    return nFrames

def input_video(video, frameRate = 10, orientation = 'landscape', displayVideo = True, displayFrames = False):
    
    frames = read_video(video, orientation, displayVideo)
    frames = select_keyframes(frames, frameRate, displayFrames)
    return frames

def save_result(image, directory, name):
    """ image is saved is 'name' in 'directory'. Unless name contains a file 
        extension, the image is saved as a png file """
    
    if not os.path.exists(directory):
        os.mkdir(directory)
           
    name = name if name.find('.') != -1 else name +'.png'
    name = os.path.join(directory, name)
    cv.imwrite(name, image)
    
def display_image(image, windowName = "Image", waitkey = 0, windowSize = (600, 800)):
    cv.namedWindow(windowName, cv.WINDOW_NORMAL)
    cv.resizeWindow(windowName, windowSize[1], windowSize[0])
    cv.moveWindow(windowName, 0, 0) 
    cv.imshow(windowName, image)
    cv.waitKey(waitkey)
    cv.destroyAllWindows()
    