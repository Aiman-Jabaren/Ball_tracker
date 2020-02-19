#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 14:26:10 2020

@author: aymanjabaren
"""


import numpy as np
import cv2
import imutils
import os
from os.path import isfile, join
 
def convert_frames_to_video(pathIn,pathOut,fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
 
    #for sorting the file names properly
    files.sort(key = lambda x: int(x[5:-4]))
 
    for i in range(len(files)):
        filename=pathIn + files[i]
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        print(filename)
        #inserting the frames into an image array
        frame_array.append(img)
 
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
 
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()
    
    
    


#location lists
x_loc = []
y_loc = []


#%%
#define paths and direcotories
output_loc = './tracking_imgs'
output_loc_mask = './tracking_imgs_mask'


test_video_path = './test_video.avi'


if not os.path.exists('./tracking_imgs'):
    os.makedirs('tracking_imgs')
if not os.path.exists('./tracking_imgs_mask'):
    os.makedirs('tracking_imgs_mask')


#frame counter
count = 0


#lowe and upper  color range 
Lower = (0, 20, 100)  
Upper = (16, 120, 200)


#import video
vs = cv2.VideoCapture(test_video_path)


#%%

#define mask_total which masks the background/ constant objects
mask_total = np.zeros((337,600))

while True:
    


    _, frame = vs.read()



    if frame is None:
        break
    
    
	# Manipulate frame size, and colors in order to extract the object
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
	
    mask = cv2.inRange(hsv, Lower, Upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    
    #Define Noise mask
    mask_total+=mask
    
    if count == 10:
        mask_total2 = mask_total > 225*8
        final_mask = 255.*mask_total2.astype(int)
   
    #zero constant noise (background) in the next mask
    if count > 10:
        mask[final_mask >= 5] = 0
    
    # find contours according to mask
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
    	cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None
    
    
    if len(cnts) > 0:
		# Extract the largest mask
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        #Ball mask has to be bigger than a certain size                
        if radius > 7:
			# draw the bounding box and append the ball location
    	    cv2.rectangle(frame, (int(x-radius/2), int(y-radius/2)), (int(x+radius/2), int(y+radius/2)),(0, 255, 255), 2)
    	    x_loc.append(x)
    	    y_loc.append(y)
        else:
            x_loc.append(None)
            y_loc.append(None)
            
    
    #write3 mask image and frame image with bounding box
    cv2.imwrite(output_loc + "/%#05d.jpg" % (count+1), frame)
    cv2.imwrite(output_loc_mask + "/mask%#05d.jpg" % (count+1), mask)
    count = count + 1
    





#%%
    
    
    
#Plot Ball location and convert images to videos
import matplotlib.pyplot as plt

fig = plt.figure()

plt.plot(x_loc,y_loc)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Ball Location')


pathIn= './tracking_imgs_mask/'
pathOut = 'video_mask.mp4'
fps = 25.0
convert_frames_to_video(pathIn, pathOut, fps)

pathOut = 'video_boundbox.mp4'
fps = 25.0
output = output_loc + '/'
#convert_frames_to_video(output, pathOut, fps)






