import cv2
import numpy as np
import time
from sys import platform
if platform == "linux" or platform == "linux2":
    cap = cv2.VideoCapture(1)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH);
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT);

    IMG_WIDTH = int(width * 0.5)
    IMG_HEIGHT = int(height * 0.5)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,IMG_WIDTH);
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,IMG_HEIGHT);
else:
    cap = cv2.VideoCapture(1)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH);
    height = cap.get(cv2.CAP_PROP_FRAME_WIDTH);

    IMG_WIDTH = int(width * 0.5)
    IMG_HEIGHT = int(height * 0.5)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,IMG_WIDTH);
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,IMG_HEIGHT);


total_area_num_dict = {'green':32, 'blue':52, 'red':22}

# set blue thresh
lower_orange = np.array([10,200,200])
upper_orange = np.array([20,255,255])

lower_bright_orange = np.array([20,150,150])
upper_bright_orange = np.array([30,255,255])

lower_green = np.array([30,40,150]) # 53, 110, 217 #99 132 177 # 54, 73, ?
upper_green = np.array([60,255,255])

lower_red1 = np.array([0,100,160]) #8, 61, 247
upper_red1 = np.array([15,255,255])

lower_red2 = np.array([165,100,160])
upper_red2 = np.array([180,255,255])

lower_yellow = np.array([20,60,190])  #27,94,252 # ? ?,225
upper_yellow = np.array([33,150,255])

lower_blue = np.array([80,50,150]) #98, 183, 226 # 104, 117, ?
upper_blue = np.array([110,255,255])

lower_dict = {'orange': lower_orange, 'green':lower_green, 'yellow':lower_yellow, 'blue':lower_blue}
upper_dict = {'orange': upper_orange, 'green':upper_green  , 'yellow':upper_yellow, 'blue':upper_blue}

center_point = None
corners = {}

PIXEL_MIN_X = None
PIXEL_MAX_X = None
PIXEL_MIN_Y = None
PIXEL_MAX_Y = None

REAL_MIN_X = -1.0 #meter
REAL_MAX_X = 1.0 #meter
REAL_MIN_Y = -1.0 #meter
REAL_MAX_Y = 1.0 #meter

def check_color(img, target_color, lower_dict, upper_dict):
    
    global lower_red1, lower_red2, upper_red1, upper_red2
    Img = img.copy()
    
    if target_color != 'red':
        #Img = img.copy()        
        # change to hsv model
        hsv = cv2.cvtColor(Img, cv2.COLOR_BGR2HSV)

        # get mask
        mask = cv2.inRange(hsv, lower_dict[target_color], upper_dict[target_color])
    
    else:
        '''Img = img.copy()
        Img[0:16,:,:] = 0
        Img[-16:,:,:] = 0
        Img[:,0:16,:] = 0
        Img[:,-16:,:] = 0'''
        # change to hsv model
        hsv = cv2.cvtColor(Img, cv2.COLOR_BGR2HSV)
    
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        #cv2.imshow('Mask1', mask1)
        
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        #cv2.imshow('Mask2', mask1)
        mask = mask1 + mask2
        
    #cv2.imshow('Mask', mask)
    dilation = cv2.GaussianBlur(mask,(5,5),0)

    # detect blue remove the non-target regions
    target = cv2.bitwise_and(Img, Img, mask=dilation)    

    # binary image
    ret, binary = cv2.threshold(dilation, 128, 255, cv2.THRESH_BINARY)

    # search contours and rank them by the size of areas
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    p = 0
    rects = []
    contours = contours[::-1]
    count_area = 0
    for i in contours:  # iterate all contours
           
        x, y, w, h = cv2.boundingRect(i)  # the coordinate of point left-top corner and length, width
        
        if w * h < 100:
            continue

        rects += [[x,y,w,h]]
        # plot the bounding box
        cv2.rectangle(Img, (x, y), (x + w, y + h), (0, 255,), 3)
        #cv2.rectangle(Img, (5, 5), (5 + w, 5 + h), (0, 255,), 3)
        # putText, not necessary
        font = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(Img, str(p), (x - 10, y + 10), font, 1, (0, 0, 255), 2)  # +- 10 for better display

    #cv2.imshow('hsv', hsv)
    #cv2.imshow('dilation', dilation)
    #cv2.imshow('target', target)
    #cv2.imshow('Mask', mask)
    #cv2.imshow("prod", dilation)
    cv2.imshow(target_color, Img)

    return Img, rects
  
def compute_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    
def get_robots(img, total_area_num_dict, lower_dict, upper_dict):
    global lower_red1, lower_red2, upper_red1, upper_red2
    
    big_boxes = []
    small_boxes = []
    Img = img.copy()
    contours_big_area = {}
    contours_small_area = {}
    contours_area = {}
    
    for target_color, value in total_area_num_dict.items():
    
        if target_color != 'red':
            # change to hsv model
            hsv = cv2.cvtColor(Img, cv2.COLOR_BGR2HSV)
            # get mask
            mask = cv2.inRange(hsv, lower_dict[target_color], upper_dict[target_color])    
        else:
            # change to hsv model
            hsv = cv2.cvtColor(Img, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            #cv2.imshow('Mask1', mask1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            #cv2.imshow('Mask2', mask1)
            mask = mask1 + mask2
        
        #cv2.imshow('Mask ' + target_color, mask)

        dilation = cv2.GaussianBlur(mask,(3,3),0)

        # detect blue remove the non-target regions
        target = cv2.bitwise_and(Img, Img, mask=dilation)    

        # binary image
        ret, binary = cv2.threshold(dilation, 128, 255, cv2.THRESH_BINARY)

        # search contours and rank them by the size of areas
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[::-1]
        contours_area[target_color] = contours[:total_area_num_dict[target_color]]
    
    robot_label = []
    robot_contour = []
    robot_data = []
    
    tolerance = 5
    
    for b_color, contours_big in contours_area.items():
        for contour_big in contours_big:
            x_big, y_big, w_big, h_big = cv2.boundingRect(contour_big)
            
            #epsilon = 0.1 * cv2.arcLength(contour_big, True)
            #approx = cv2.approxPolyDP(contour_big, epsilon, True)
            
            if w_big * h_big < 250:
                continue
            
            for s_color, contours_small in contours_area.items():
                for contour_small in contours_small:
                    x_small, y_small, w_small, h_small = cv2.boundingRect(contour_small)
                    
                    if x_big == x_small and y_big == y_small or b_color == s_color:
                        continue
                    
                    if (x_big - tolerance < x_small and x_big + w_big + tolerance > x_small + w_small) and \
                       (y_big - tolerance < y_small and y_big + h_big + tolerance > y_small + h_small):
                        robot_label += [b_color[0] + s_color[0]]
                        robot_contour += [(x_big, y_big, w_big, h_big, x_small, y_small, w_small, h_small)]
                        direction_vector_x = (x_small+w_small/2) - (x_big+w_big/2)
                        direction_vector_y = (y_small+h_small/2) - (y_big+h_big/2)
                        norm = np.sqrt(direction_vector_x ** 2 + direction_vector_y ** 2)
                        direction_vector_x /= norm
                        direction_vector_y /= norm
                        
                        robot_data += [(x_big + w_big / 2, (y_big + h_big / 2), direction_vector_x, direction_vector_y)]
    
    for idx, rect in enumerate(robot_contour):
        x,y,w,h,x2,y2,w2,h2=rect
        cv2.rectangle(Img, (x, y), (x + w, y + h), (0, 255,), 3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(Img, robot_label[idx], (x - 14, y - 14), font, 0.7, (0, 0, 255), 2)  # +- 10 for better display
        lineThickness = 2
        cv2.line(Img, (int(robot_data[idx][0]), int(robot_data[idx][1])), (int(robot_data[idx][0] + robot_data[idx][2] * 30), int(robot_data[idx][1] + robot_data[idx][3] * 30)), (0,255,0), lineThickness)
    
    if __name__ == "__main__":    
        cv2.imshow('robot', Img)

    return Img, robot_label, robot_data
        
#for key, value in corners.items():
#    print(tuple(value))
#    cv2.rectangle(Img, tuple(value), tuple(value + 5), (0, 255, 255, 0), 3)

def pixel2coordinate(x, y, PIXEL_MIN_X, PIXEL_MAX_X, PIXEL_MIN_Y, PIXEL_MAX_Y, REAL_MIN_X, REAL_MAX_X, REAL_MIN_Y, REAL_MAX_Y):

    #print(x, y, PIXEL_MIN_X, PIXEL_MAX_X, PIXEL_MIN_Y, PIXEL_MAX_Y, REAL_MIN_X, REAL_MAX_X, REAL_MIN_Y, REAL_MAX_Y)
    
    coor_x = REAL_MAX_X - (REAL_MAX_X - REAL_MIN_X) * (PIXEL_MAX_X - x) / (PIXEL_MAX_X - PIXEL_MIN_X)
    coor_y = REAL_MAX_Y - (REAL_MAX_Y - REAL_MIN_Y) * (PIXEL_MAX_Y - y) / (PIXEL_MAX_Y - PIXEL_MIN_Y)
    
    return (coor_x, coor_y)

if __name__ == "__main__":
    timestep = 0.5
    current_time = time.time()    
    while True:
        if (time.time() - current_time) > timestep:
            current_time = time.time()
            ret, img = cap.read()

            #cv2.imshow('Capture', img)
            img = img[PIXEL_MIN_Y:PIXEL_MAX_Y, PIXEL_MIN_X:PIXEL_MAX_X]

            file = open("robots_info.txt","a")
            if True:        
                _, robot_label, robot_data = get_robots(img, total_area_num_dict, lower_dict, upper_dict)
                cv2.imwrite('videos/%f.png' %current_time, _)
                for idx, _ in enumerate(robot_data):
                    x_center, y_center, x_direction, y_direction = robot_data[idx]
                
                    print("time, robot_label, x_center, y_center, x_direction, y_direction")
                    print(current_time, robot_label[idx], x_center, IMG_HEIGHT - y_center, x_direction, -y_direction)
                    data = (current_time, robot_label[idx], x_center, IMG_HEIGHT - y_center, x_direction, -y_direction)
                    buffer = " ".join(map(str, data))
                    file.write(buffer+"\n")

            file.close()    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break