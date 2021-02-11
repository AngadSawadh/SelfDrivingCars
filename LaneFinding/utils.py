import numpy as np

def extend_lines(lines,ybottom):
    """ Input:- 3d ndarray where 1st dimension represent various lines and second dimension has two points lying on that line [x1,y1,x2,y2]
        Output:- 3d ndarray with new x1,y1,x2,y2
        extends the lines to the bottom of the image
    """
    extended_lines = []

    if lines.shape[0] < 2:
        return lines
    for line in lines:
        for x1,y1,x2,y2 in line:
            if y2<=y1:
                dx = x2-x1
                dy = y2-y1
                x_intercept = np.round((ybottom-y2)*dx/dy + x2) 
                temp = [[x_intercept,ybottom,x2,y2]]
                extended_lines.append(temp)
            else:
                dx = x2-x1
                dy = y2-y1
                x_intercept = np.round((ybottom-y2)*dx/dy + x2)
                temp = [[x_intercept,ybottom,x1,y1]]
                extended_lines.append(temp)
    return np.array(extended_lines,dtype=np.int32)



def lanelinefilter(lines,ybottom,width):
    """ Input:- 2d ndarray where 1st dimension represent various lines and second dimension has two points lying on that line [x1,y1,x2,y2]
        Output:- reduced 2d ndarray with lines having y_bottom intercept within the image
    """
    good_lines = []


    if lines.shape[0] < 2:
        return lines
    for line in lines:
        for x1,y1,x2,y2 in line:
            dx = x2-x1
            dy = y2-y1
            x_intercept = (ybottom-y2)*dx/dy + x2
        
        if x_intercept<=width and x_intercept>=0:
            good_lines.append(line)

    return np.array(good_lines)

def slopefilter(lines):
    """
    """
    sloped_lines = []
    if lines.shape[0] < 2:
        return lines
    for line in lines:
        for x1,y1,x2,y2 in line:
            dx = x2-x1
            dy = y2-y1
            if dx!=0:
                if 0.2<=np.abs(dy/dx)<=3:
                    sloped_lines.append(line)
    return np.array(sloped_lines)            


def interpolate(lines,xsize,ysize):
    """
    """
    neg_slope = []
    pos_slope = []
    neg_bias = []
    pos_bias = []
    final_lane_line = []

    if lines.shape[0] < 2:
        return lines

    for line in lines:
        for x1,y1,x2,y2 in line:
            dx = x2-x1
            dy = y2-y1
            if dy/dx < 0:
                neg_slope.append(dy/dx)
                neg_bias.append(-x2*dy/dx + y2)
            else:
                pos_slope.append(dy/dx)
                pos_bias.append(-x2*dy/dx + y2)

    
    #print(pos_slope)
    #print(neg_slope)
    #print(pos_bias)
    #print(neg_bias)
    slope_left_lane = np.median(neg_slope)
    slope_right_lane = np.median(pos_slope)
    bias_left_lane = np.median(neg_bias)
    bias_right_lane = np.median(pos_bias)
    #print(slope_left_lane)
    #print(slope_right_lane)
    #print(bias_left_lane)
    #print(bias_right_lane)

    final_lane_line.append([[(ysize-bias_left_lane)/slope_left_lane ,ysize ,(3*ysize/5-bias_left_lane)/slope_left_lane ,3*ysize/5]])
    final_lane_line.append([[(ysize-bias_right_lane)/slope_right_lane ,ysize ,(3*ysize/5-bias_right_lane)/slope_right_lane ,3*ysize/5]])

    return np.array(final_lane_line,dtype=np.int32)

def interpolate_avearage(lines,y_bottom):
    x_down_left = []
    x_down_right = []
    x_top_left = []
    x_top_right = []
    y_top_left = []
    y_top_right = []
    final_lines = []

    for line in lines:
        for x1,y1,x2,y2 in line:
            dx = x2-x1
            dy = y2-y1
            if dy/dx < 0:
                if y1<y2:
                    y_top_left.append(y1)
                    x_top_left.append(x1)
                    x_down_left.append(x2)

                else:
                    y_top_left.append(y2)
                    x_top_left.append(x2)
                    x_down_left.append(x1)

            else:
                if y1<y2:
                    y_top_right.append(y1)
                    x_top_right.append(x1)
                    x_down_right.append(x2)

                else:
                    y_top_right.append(y2)
                    x_top_right.append(x2)
                    x_down_right.append(x1)
    
    y_top_left_final = np.mean(y_top_left)
    y_top_right_final = np.mean(y_top_right)
    x_top_left_final = np.mean(x_top_left)
    x_top_right_final = np.mean(x_top_right)
    x_down_left_final = np.mean(x_down_left)
    x_down_right_final =  np.mean(x_down_right)

    final_lines.append([[x_down_left_final,y_bottom,x_top_left_final,y_top_left_final]])
    final_lines.append([[x_down_right_final,y_bottom,x_top_right_final,y_top_right_final]])
            
    return np.array(final_lines,dtype=np.int32)


    
