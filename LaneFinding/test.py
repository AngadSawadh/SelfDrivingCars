import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import utils

def lanefind(image):
    #image  = mpimg.imread(r"G:\SelfDrivingcarsPractice\LaneFinding\data\images\solidWhiteRight.jpg")
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    ysize,xsize = gray.shape

    #apply the gaussian blur
    kernel_size = 5
    blurred_gray = cv2.GaussianBlur(gray,(kernel_size,kernel_size),0)

    #apply the canny edge detection
    high_threshold = 150
    low_threshold = 100
    edges = cv2.Canny(blurred_gray,low_threshold,high_threshold)

    #get the region of interest
    mask = np.zeros_like(edges)
    ignore_mask_color = 1
    vertices = np.array([[(0,ysize),(5*xsize/11,ysize/2),(6*xsize/11,ysize/2),(xsize,ysize)]],dtype=np.int32)
    cv2.fillPoly(mask,vertices,ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)

    #applying Hough transform to draw lines
    rho = 1
    theta = np.pi/180
    threshold = 20
    min_line_length = 30
    max_line_gap = 20
    line_image = np.zeros_like(image)

    lines = cv2.HoughLinesP(masked_edges,rho,theta,threshold,np.array([]),min_line_length,max_line_gap)
        
    filtered_lines_0 = utils.slopefilter(lines)
    filtered_lines_1 = utils.lanelinefilter(filtered_lines_0,ysize,xsize)
    filtered_lines_2 = utils.extend_lines(filtered_lines_1,ysize)
    final_lines = utils.interpolate(filtered_lines_2,xsize,ysize)

    for line in final_lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)


        
    color_edges = np.dstack((masked_edges, masked_edges, masked_edges)) 
    combo = cv2.addWeighted(image, 1, line_image, 0.3, 0) 

    return combo