# -*- coding: utf-8 -*-
"""
Created on Tue May  5 12:45:41 2020

@author: donal
"""

import cv2 as cv
import numpy as np
import utils

def detect_keypoints(frames, kpDetector = "SIFT", xPadding = 1, yPadding = 3, display = False):
    """
    Parameters
    ----------
    frames : numpy array
        frames in which to detect keypoints.
    detector : string
        algorithm to use for detecting keypoints.

    Returns
    -------
    None.

    """
    m, x, y, z  = frames.shape
    n = int(m/2)
    #dictionaries to hold keypoints and their descriptors.
    kp = {} #key point locators
    desc = {} #key point descriptors
    
    #convert images to grayscale
    gray = np.zeros((m, x, y), dtype = np.uint8)
    for i in range(m):
        gray[i] = cv.cvtColor(frames[i],cv.COLOR_BGR2GRAY)
    
    #create keypoint detector object according to chosen algorithm
    if kpDetector == "SIFT":
        detector = cv.xfeatures2d.SIFT_create(nfeatures = 0, nOctaveLayers = 3, contrastThreshold = 0.04, edgeThreshold = 10, sigma = 1.6)
    if kpDetector == "SURF":
        detector = cv.xfeatures2d.SURF_create(hessianThreshold = 500, nOctaves = 4, nOctaveLayers = 2, extended = True, upright = False)
    if kpDetector == "ORB":
        detector = cv.ORB_create(nfeatures = 15000, scaleFactor = 1.2, nlevels = 8, edgeThreshold = 31, firstLevel = 0, WTA_K = 2, scoreType = cv.ORB_HARRIS_SCORE, patchSize = 31, fastThreshold = 20)
    
    #find keypoints and descriptors for each frame
    print("\nSearching for keypoints using {} algorithm . . .\n".format(kpDetector))
    for i in range(m):
        kp[i], desc[i] = detector.detectAndCompute(gray[i],None)

    #display grayscale images with keypoints
    if display:    
        for i in range(m):
            utils.display_image(cv.drawKeypoints(gray[i], kp[i], None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS),'keyPoints'+str(i), 0, (int(x/2), int(y/2)))

    #replace keypoints for centre image once in buffer 
    centre = cv.copyMakeBorder(frames[n], xPadding * x, xPadding * x,  yPadding *y, yPadding * y,cv.BORDER_CONSTANT)
    kp[n], desc[n] = detector.detectAndCompute(centre,None)  
        
    return kp, desc
    
def match_keypoints(kp, desc, frames, kpDetector = "SIFT", matchingAlgorithm = "BF", display = False):
    """
    

    Parameters
    ----------
    kp : dictionary
        kp[i] is the list of key-points for frame i.
    desc : dictionary
        desc[i] is the list of key-point descriptors for frame i.
    frames : numpy array
        numpy array of video frames - only used for displaying matches.
    kpDetector : string, optional
        Algorithm used to detect the key-points. The default is "SIFT".
    matchingAlgorithm : string, optional
        algorithm used to find matches. The default is "BF".
    display : boolean, optional
        whether or not to display the matches.

    Returns
    -------
    matches : dictionary
        matches[i] is list of matches between frames i and i+1.

    """
    
    m, x, y = frames.shape[:3]
    
    if (matchingAlgorithm == "BF") and ((kpDetector == "SIFT") or (kpDetector == "SURF")) :
        matcher = cv.BFMatcher_create(normType = cv.NORM_L2, crossCheck = False)
            
    elif (matchingAlgorithm == "BF") and (kpDetector == "ORB") :
        matcher = cv.BFMatcher_create(normType = cv.NORM_HAMMING, crossCheck = False)
            
    elif (matchingAlgorithm == "FLANN") and ((kpDetector == "SIFT") or (kpDetector == "SURF")) :  
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 100)
        matcher = cv.FlannBasedMatcher(index_params,search_params)
    elif (matchingAlgorithm == "FLANN") and (kpDetector == "ORB") :
        FLANN_INDEX_LSH = 6
        index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
        matcher = cv.FlannBasedMatcher(index_params)

    matches = {}
    
    print("\nSearching for matches using {} algorithm . . .\n".format(matchingAlgorithm))
    for i in range(m-1):
        matches[i] = matcher.knnMatch(desc[i+1], desc[i], k = 2)
        good = []
        for q,r in matches[i]:
            if q.distance < 0.5 * r.distance: #switched from 0.75
                good.append(q) 
        good = sorted(good, key = lambda x:x.distance) #added in to sort so closest match is first
        matches[i] = good
        
    if display:
       for i in range(m-1):
           utils.display_image(cv.drawMatches(frames[i+1], kp[i+1], frames[i], kp[i], matches[i][:50], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS),'matches'+str(i+1)+str(i), 0, (int(x/2), y))

    return matches
    

    