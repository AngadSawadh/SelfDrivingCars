import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import utils

def pipeline(image):
    #image  = cv2.imread(r"G:\SelfDrivingcarsPractice\LaneFinding\data\images\solidWhiteRight.jpg")
    ysize = image.shape[0]
    xsize = image.shape[1]
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    kernel_size = 5
    blurred_gray = cv2.GaussianBlur(gray,(kernel_size,kernel_size),0)

    high_threshold = 90
    low_threshold = 30
    edges = cv2.Canny(blurred_gray,low_threshold,high_threshold)

    mask = np.zeros_like(edges)
    ignore_mask_color = 255
    vertices = np.array([[(0,ysize),(5*xsize/11,ysize/2),(6*xsize/11,ysize/2),(xsize,ysize)]],dtype=np.int32)
    cv2.fillPoly(mask,vertices,ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)

    rho = 1
    theta = np.pi/180
    threshold = 60
    min_line_length = 100
    max_line_gap = 50
    line_image = np.zeros_like(image)

    lines = cv2.HoughLinesP(masked_edges,rho,theta,threshold,np.array([]),min_line_length,max_line_gap)
    filtered_lines_0 = utils.slopefilter(lines)
    filtered_lines_1 = utils.lanelinefilter(filtered_lines_0,ysize,xsize)
    filtered_lines_2 = utils.extend_lines(filtered_lines_1,ysize)
    filtered_lines_3 = utils.interpolate(filtered_lines_2,xsize,ysize)

    for line in filtered_lines_3:
            for x1,y1,x2,y2 in line:
                cv2.line(line_image,(x1,y1),(x2,y2),(0,0,255),10)

    combo = cv2.addWeighted(image, 1, line_image, 0.3, 0) 
    return combo

#cv2.imshow("Image",combo)
#cv2.waitKey(0) 
#cv2.destroyAllWindows() 
