import numpy as np
from matplotlib import pyplot as plt
import scipy.ndimage as ndimage
import cv2
import heapq
import math
import scipy
from sklearn.cluster import KMeans
import timeit
from skimage.feature import blob_doh

Plot = False
save = False

# TODO: say that you cant anything (camera covered)

def initialization():
    start = timeit.default_timer()
    map_path_offline = r"C:\Users\Carl\Desktop\Project\Maps\map6.jpg"
    map = cv2.imread(map_path_offline)
    corner_mask_path = r"C:\Users\Carl\Desktop\Project\mask.jpeg"
    mask = cv2.imread(corner_mask_path)
    dimension_paper = [118.9,84.1] #cm A0
    dim = (int(dimension_paper[1]*2),int(dimension_paper[0]*2))
    # Switching red and blue channels
    map[:, :, [0, 2]] = map[:, :, [2, 0]]
    mask[:, :, [0, 2]] = mask[:, :, [2, 0]]

    obstacles = get_obstacles(map,mask)
    thymio_coord = get_thymio_info(map,mask) # Do these online, and feed info to kalman filter
    endpoint_coord = get_endpoint_info(map,mask)
    surgically_enhanced_obstacles = process_obstacles(obstacles)
    if save:
        plt.imsave('obstacles.jpg',obstacles)
    stop = timeit.default_timer()
    print('Time: ', stop - start)

def get_obstacles(map,mask):
    corner_location = corner_detection(map,mask)
    img_straighten, M = four_point_transform(map, corner_location)
    obstacles = black_contours(img_straighten,0.001)

    return obstacles

def get_thymio_info(map,mask):
    corner_location = corner_detection(map,mask)
    img_straighten, M = four_point_transform(map, corner_location)
    thymio_map = color_filtering(img_straighten,"blue")
    thymio_coords = end_point_start_point(thymio_map, 0.001, "thymio")
    print("Thymio Coordinates + Orientation: ", thymio_coords)
    return thymio_coords

def get_endpoint_info(map,mask):
    corner_location = corner_detection(map,mask)
    img_straighten, M = four_point_transform(map, corner_location)
    endpoint_map = color_filtering(img_straighten,"green")
    endpoint_coord = end_point_start_point(endpoint_map, 0.001, "endpoint")
    print("Endpoint Coordinates: ", endpoint_coord)
    return endpoint_coord

def process_obstacles(img):
    output = np.copy(img)
    kernel = np.ones((3,3), np.uint8)
    output = cv2.erode(output, kernel, iterations=1)
    output = cv2.dilate(output, kernel, iterations=2)
    kernel2 = np.ones((9,9), np.uint8)
    output = cv2.dilate(output, kernel2, iterations=2)
    if Plot:
        plt.imshow(output)
        plt.show()
    return output

def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)

    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    if Plot:
        plt.imshow(warped)
        plt.show()
    # return the warped image
    return warped, M

def corner_detection(img,mask):
    large_image = np.copy(img)
    small_image = np.copy(mask)
    method = cv2.TM_CCOEFF_NORMED
    large_image = color_filtering(large_image,"red")
    corner_location = np.zeros((4,2))
    large_image = cv2.cvtColor(large_image, cv2.COLOR_RGB2GRAY)
    small_image = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    _, large_image = cv2.threshold(large_image, 30, 255, cv2.THRESH_BINARY)

    for i in range (0,4):
        result = cv2.matchTemplate(small_image, large_image, method)
        mn,_,mnLoc,_ = cv2.minMaxLoc(result) # We want the minimum squared difference
        MPx,MPy = mnLoc # Extract the coordinates of our best match
        trows,tcols = small_image.shape[:2] # Get the size of the template. This is the same size as the match.
        cv2.rectangle(large_image, (MPx,MPy),(MPx+tcols,MPy+trows),(0,0,255),2) # Draw the rectangle on large_image
        new_img = 255*np.ones(large_image.shape,np.uint8)
        large_image[MPy:MPy+trows,MPx:MPx+trows] = new_img[MPy:MPy+trows,MPx:MPx+trows]
        small_image = cv2.rotate(small_image, cv2.ROTATE_90_CLOCKWISE)
        if i == 0:
            corner_location[i,:] = [MPy,MPx]
        elif i == 1:
            corner_location[i,:] = [MPy,MPx+tcols]
        elif i == 2:
            corner_location[i,:] = [MPy+trows,MPx+tcols]
        else:
            corner_location[i,:] = [MPy+trows,MPx]
    corner_location = np.fliplr(corner_location)

    return corner_location

def color_filtering(img,color):
    large_image = np.copy(img)
    method = cv2.TM_SQDIFF_NORMED
    if color == "red":
        lower = np.array([100,0,0])
        upper = np.array([255,100,100])
    elif color == "blue":
        lower = np.array([0,0,70])
        upper = np.array([90,90,255])
    elif color == "green":
        lower = np.array([0,90,0])
        upper = np.array([90,255,90])
    elif color == "black":
        large_image = cv2.cvtColor(large_image, cv2.COLOR_RGB2GRAY)
        _,output = cv2.threshold(large_image,10,255,cv2.THRESH_BINARY)
        if Plot:
            plt.imshow(output)
            plt.show()
        return output
    color_mask = cv2.inRange(large_image, lower, upper)
    large_image = cv2.bitwise_and(large_image, large_image, mask=color_mask)
    if Plot:
        plt.imshow(large_image)
        plt.show()
    return large_image

def black_contours(img,constant):
    # obstacles = find_contours(img_straighten_grey, 0.001) #OR
    output = np.copy(img) #copy the image
    output_grey = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY) # convert to gray
    output_grey = output_grey.astype(np.uint8) #uint8 type to use as binary image
    _, threshold = cv2.threshold(output_grey, 100, 255, cv2.THRESH_BINARY) # threshold and obtain 0 and 255 values only
    contours,_ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # find contours in the image
    output_grey_contours = np.zeros([output_grey.shape[0],output_grey.shape[1]]) # get a placeholder image for the contours

    largest_areas = sorted(contours, key=cv2.contourArea) # sort the contours from smallest to largest
    largest_areas = largest_areas[:-1]
    largest_areas = largest_areas[::-1] # flip the array and make it largest to smallest
    for cnt in largest_areas[:3]:
        approx = cv2.approxPolyDP(cnt, constant*cv2.arcLength(cnt, True), True) # Approximate the contour(s)
        cv2.drawContours(output_grey_contours, [approx], 0, (255), thickness=cv2.FILLED) #draw the contour(s) on the place holder  # replace thickness=cv2.FILLED with thickness=5 for edges only
        x = approx.ravel()[0] # get x coordinate of contour point
        y = approx.ravel()[1] # get y coordinate of contour point
    if Plot:
        plt.imshow(output_grey_contours)
        plt.show()
    return output_grey_contours

def end_point_start_point(img,constant,point,clust=1):
    # obstacles = find_contours(img_straighten_grey, 0.001) #OR
    output = np.copy(img) #copy the image
    output_grey = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY) # convert to gray
    output_grey = output_grey.astype(np.uint8) #uint8 type to use as binary image
    _, threshold = cv2.threshold(output_grey, 1, 255, cv2.THRESH_BINARY) # threshold and obtain 0 and 255 values only
    contours,_ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # find contours in the image
    output_grey_contours = np.zeros([output_grey.shape[0],output_grey.shape[1]]) # get a placeholder image for the contours

    largest_areas = sorted(contours, key=cv2.contourArea) # sort the contours from smallest to largest
    if not clust: # if we are clustering, the outer boarder is removed automatically
        largest_areas = largest_areas[:-1] # remove the outer border of the picture
    largest_areas = largest_areas[::-1] # flip the array and make it largest to smallest
    if (point == "thymio"):
        largest_areas = largest_areas[:2] # keep only the two largest contours corresponding to the interesting parts (and remove the noisy outputs)
        coordinates = np.zeros((2,2)) # place holder for the coordinates
    elif(point == "endpoint"):
        largest_areas = largest_areas[:1] # keep only the largest contour (endpoint)
        coordinates = np.zeros((1,2)) # place holder for the coordinates
    for cnt in largest_areas:
        approx = cv2.approxPolyDP(cnt, constant*cv2.arcLength(cnt, True), True) # Approximate the contour(s)
        cv2.drawContours(output_grey_contours, [approx], 0, (255), thickness=cv2.FILLED) #draw the contour(s) on the place holder  # replace thickness=cv2.FILLED with thickness=5 for edges only
        x = approx.ravel()[0] # get x coordinate of contour point
        y = approx.ravel()[1] # get y coordinate of contour point
    location_image = np.zeros([output_grey.shape[0],output_grey.shape[1]]) # Place holder for the
    i = 0 # index
    for c in largest_areas:
        M = cv2.moments(c) # calculating moments for each contour, i.e center of the circle englobing the contours
        cX = int(M["m10"] / M["m00"]) # calculate x coordinate of center
        cY = int(M["m01"] / M["m00"]) # calculate y coordinate of center
        cv2.circle(location_image, (cX, cY), 5, (255, 255, 255), -1) # Draw the circle englobing the contours
        if(i<2):
            coordinates[i] = [cX, cY]
        else:
            coordinates = [cX, cY]
        i = i + 1
    if Plot:
        plt.imshow(location_image)
        plt.show()
    if (point == "thymio"):
        thymio_center_coord, thymio_orientation = orientation_location_thymio(output_grey, coordinates) # Get coordinates and orientation of the center of the thymio
        coords = [thymio_center_coord, thymio_orientation] # Concatinate into one variable
    elif(point == "endpoint"):
        coords = coordinates # No need to do further action. The center of the circle englobing the endpoint is the center of the endpoint
    return coords

def orientation_location_thymio(img,coordinates_thymio):

    center_thymio = [(coordinates_thymio[0][0]+coordinates_thymio[1][0])/2, (coordinates_thymio[0][1] + coordinates_thymio[1][1])/2] # Getting x and y of image
    img_x,img_y = img.shape[:2] # x and y dimentions of the image
    orientation_rad = np.arctan2(img_y-center_thymio[0], img_x-center_thymio[1]) # get angle with point (0,1), TODO: We have a constant offset of 8 degrees?? wurrup wid dat bish?
    orientation_deg = orientation_rad*180/math.pi # convert rad to degree
    return center_thymio, orientation_deg



if __name__ == "__main__":
    initialization()
