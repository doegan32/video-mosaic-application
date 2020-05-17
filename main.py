# -*- coding: utf-8 -*-
"""
Created on Tue May  5 12:43:30 2020

@author: donal
"""


# import libraries
import argparse
import os
# import custom made modules
import utils
import features
import homography

def parse_args():
    parser = argparse.ArgumentParser(description = "Create panorma images from your videos")
    parser.add_argument('inputVideo', type = str, help = "name of input video to be processed. This argument must be provided")
    parser.add_argument('--inputDir', default = 'input', type = str, help = "directory in which to find the input video")
    parser.add_argument('--saveOff', action = 'store_true', help = "use this command if you do not want to save the output")
    parser.add_argument('--saveAs', type = str, help = "name of final saved panorama")
    parser.add_argument('--saveDir', default = 'output', type =str, help = "directory in which to save panorama")
    parser.add_argument('--xPadding', default = 0, type = int, help = 'To determine size of panorama. xPadding*frameHeight will be added above and below')
    parser.add_argument('--yPadding', default = 3, type = int, help = 'To determine size of panorama. yPadding*frameWidth will be added left and right')
    parser.add_argument('--frameRate', default = 10, type = int, help = "If frameRate = N, every Nth frame will be used")
    parser.add_argument('--orientation', default = 'lanscape', type = str, help = 'orientation of camera', choices = ['landscape', 'portrait'])
    parser.add_argument('--kpDetector', default = 'SIFT', type = str, help = 'algorithm for finding keypoints', choices = ['SIFT', 'SURF', 'ORB'])
    parser.add_argument('--matcher', default = 'BF', type = str, help = 'algorithm for finding matches', choices = ['BF', 'ORB', 'FLANN'])
    parser.add_argument('--displayVideoOff', action = 'store_false', help = 'use this command to turn off display of video to be processed')
    parser.add_argument('--displaySteps', action = 'store_true', help = 'turn on display of intermediate steps, e.g. show keypoints and matches')
    return parser.parse_args()

def main():
    
    """
    main function for creating panorama images
    """
    # read in command line arguments
    args = parse_args()
    
    # check if input video exists and if so return the selected frames for stitching
    path = os.path.join(args.inputDir, args.inputVideo)
    if os.path.exists(path):
        frames = utils.input_video(path, args.frameRate, args.orientation, args.displayVideoOff, args.displaySteps)
    else:
        raise RuntimeError("Could not find input video. . . ")
    
    # identify keypoints and construct their descriptors
    kp, desc = features.detect_keypoints(frames, args.kpDetector, args.xPadding, args.yPadding, args.displaySteps)
    # identify matches between adjacent images
    matches = features.match_keypoints(kp, desc, frames, args.kpDetector, args.matcher, args.displaySteps)
    # calculate homography transformations
    H = homography.RANSAC_Homographies(matches, kp, args.displaySteps)
    # stitch to create final panorama
    panorama = homography.panorama(H, frames, args.xPadding, args.yPadding, args.displaySteps)
    
    
    # save panorama if desired and creating directory if needed
    if not args.saveOff:
        if args.saveAs is not None:
            utils.save_result(panorama, args.saveDir, args.saveAs)
        else:
            utils.save_result(panorama,  args.saveDir, os.path.splitext(args.inputVideo)[0])
    
    # display final panorama
    utils.display_image(panorama, "final panorama", 0, (400, 1600))
        



if __name__ == '__main__':
    main()