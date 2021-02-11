import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import utils

#Helper Function


def lane_detection_pipeline(image):
    """ Takes a RGB image as input,
        and gives an RGB image with lines drawn on the lane lines.
        
        Input is a numpy nd array of dimension [x,y,3]"""

    #we first convert our image to a grayscale image
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    ysize,xsize = gray.shape
    plt.imshow(gray,cmap='Greys_r')
    plt.show()

    #apply the gaussian blur
    kernel_size = 5
    blurred_gray = cv2.GaussianBlur(gray,(kernel_size,kernel_size),0)

    #apply the canny edge detection
    high_threshold = 100
    low_threshold = 50
    edges = cv2.Canny(blurred_gray,low_threshold,high_threshold)
    plt.imshow(edges,cmap='Greys_r')
    plt.show()

    #get the region of interest
    mask = np.zeros_like(edges)
    ignore_mask_color = 1
    vertices = np.array([[(0,ysize),(5*xsize/11,ysize/3),(6*xsize/11,ysize/3),(xsize,ysize)]],dtype=np.int32)
    cv2.fillPoly(mask,vertices,ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)
    plt.imshow(masked_edges,cmap='Greys_r')
    plt.show()

    #applying Hough transform to draw lines
    rho = 1
    theta = np.pi/180
    threshold = 1
    min_line_length = 10
    max_line_gap = 1
    line_image = np.zeros_like(image)

    lines = cv2.HoughLinesP(masked_edges,rho,theta,threshold,np.array([]),min_line_length,max_line_gap)
    
    filtered_lines = utils.slopefilter(lines)
    #filtered_line = utils.lanelinefilter(lines,ysize,xsize)
    #filtered_line_1 = utils.extend_lines(filtered_line,ysize)
    #final_lines = utils.interpolate(filtered_line_2,xsize,ysize)

    for line in filtered_lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    
    color_edges = np.dstack((masked_edges, masked_edges, masked_edges)) 
    combo = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0) 
    plt.imshow(combo)
    plt.show()

image  = mpimg.imread(r"G:\SelfDrivingcarsPractice\LaneFinding\data\images\solidYellowCurve.jpg")
print("The image is of shape:",image.shape," of type:",type(image))
lane_detection_pipeline(image)





