# -*- coding: utf-8 -*-
"""
Created on Wed May  6 15:35:38 2020

@author: donal
"""
import numpy as np
import cv2 as cv
import utils

def RANSAC_Homographies(matches, kp, display = False):
    """
    Parameters
    ----------
    matches : TYPE
        DESCRIPTION.
    kp : TYPE
        DESCRIPTION.

    Returns
    -------
    H : numpy array of dimension (m,3,3)
        DESCRIPTION.

    """
    
    m = len(kp)  #number of frames being used
    n = int(m/2) #index of middle frame
    
    H = np.float64(np.zeros((m,3,3))) #buffer to hold homograpy matrices
    H[n] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) #initialize long-term homography to the identity
    
    print("\nCalculating homographies using RANSAC. . . .\n")
    
    #move symmetrically from centre calculating long term homographies to centre frame
    for i in range(n):
        src = np.float32([ kp[n     + i][z.trainIdx].pt for z in matches[n + i] ]).reshape(-1,1,2)
        dst = np.float32([ kp[n + 1 + i][z.queryIdx].pt for z in matches[n + i] ]).reshape(-1,1,2)
        H[n + 1 + i], masked = cv.findHomography(dst, src, cv.RANSAC, 5.0) #frame-to-frame homography
        H[n + 1 + i] = np.dot(H[n + i], H[n + 1 + i] ) #long-term homography
        
    
        src = np.float32([ kp[n - 1 - i][z.trainIdx].pt for z in matches[n - 1 - i] ]).reshape(-1,1,2)
        dst = np.float32([ kp[n -     i][z.queryIdx].pt for z in matches[n - 1 - i] ]).reshape(-1,1,2)
        H[n - 1 - i], masked = cv.findHomography(src, dst, cv.RANSAC, 5.0) #frame-to-frame homography
        H[n - 1 - i] = np.dot(H[n - i], H[n - 1 - i] ) #long-term homography

    #if display:
    for i in range(m):
        print("\nHomography H_{:d},{:d} is \n".format(i,n), H[i]  )
    
    return H
    
def panorama(H, frames, xPadding = 1, yPadding = 3, display = True):
    """
    Parameters
    ----------
    H : TYPE
        DESCRIPTION.
    frames : TYPE
        DESCRIPTION.
    centre : TYPE
        DESCRIPTION.

    Returns
    -------
    int
        DESCRIPTION.

    """
    m, x, y, z = frames.shape
    n = int(m/2) #index of centre frame

    #create large buffer for panorama
    #centre = np.uint8(np.vstack((np.zeros((x * Xpadding, 2 * y * Ypadding + y, z)), 
     #                            np.uint8(np.hstack((np.zeros((x, y * Ypadding, z)), 
       #                     frames[n], np.zeros((x, y * Ypadding, z))))), np.zeros((x * Xpadding, 2 * y * Ypadding + y, z)))))
    
    centre = cv.copyMakeBorder(frames[n], xPadding * x, xPadding * x,  yPadding *y, yPadding * y,cv.BORDER_CONSTANT)
    
    s, t = centre.shape[:2]
    
    print("\nStiching images together. . . .\n")
    
    
    for i in range(n):
        fromim_tR = cv.warpPerspective(frames[n-1-i], H[n-1-i], (t, s)) #warp first side
        fromim_tL = cv.warpPerspective(frames[n+1+i], H[n+1+i], (t, s)) #warp second side
 
        #alpha = np.any(centre > 0, axis = 2 ) #indicate whether or not pixel already filled
        #alpha = np.stack((alpha, alpha, alpha), axis = 2)
        #centre = np.uint8(fromim_tR * (1-alpha) + centre * alpha + fromim_tL * (1-alpha))
        
        #alpha = np.any(fromim_tR > 0, axis = 2)
        #alpha = np.stack((alpha, alpha, alpha), axis = 2)
        #centre[:,:,:] = fromim_tR[:,:,:] * alpha + centre[:,:,:] * (1 - alpha)
        
        #alpha = np.any(fromim_tL > 0, axis = 2)
        #alpha = np.stack((alpha, alpha, alpha), axis = 2)
        #centre[:,:,:] = fromim_tL[:,:,:] * alpha + centre[:,:,:] * (1 - alpha)
        
        
        #NEXT BLOCK OF CODE IS ALPHA BLEND
        ret, maskW = cv.threshold(fromim_tR, 0,1, cv.THRESH_BINARY)
        ret, maskC = cv.threshold(centre, 0, 1, cv.THRESH_BINARY)
        maskO = np.multiply(maskW, maskC)
        maskC = cv.subtract(maskC, maskO)
        maskW = cv.subtract(maskW, maskO)
        centre = np.multiply(centre, maskC) + 0.5 * np.multiply(centre, maskO)+np.multiply(fromim_tR, maskW)+ 0.5 * np.multiply(fromim_tR, maskO)
        centre = np.uint8(centre)
        ret, maskW = cv.threshold(fromim_tL, 0,1, cv.THRESH_BINARY)
        ret, maskC = cv.threshold(centre, 0, 1, cv.THRESH_BINARY)
        maskO = np.multiply(maskW, maskC)
        maskC = cv.subtract(maskC, maskO)
        maskW = cv.subtract(maskW, maskO)
        centre = np.multiply(centre, maskC) + 0.5 * np.multiply(centre, maskO)+np.multiply(fromim_tL, maskW)+ 0.5 * np.multiply(fromim_tL, maskO)
        centre = np.uint8(centre)
        
        if display: 
            utils.display_image(centre, 'panorama', 0, (400, 1600))

    return centre
