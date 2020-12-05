import numpy as np
from matplotlib import pyplot as plt
import cv2
import math
import timeit
from skimage import exposure

Plot = False
save = False

def initialization():
    start = timeit.default_timer()
    map_path_offline = r"C:\Users\Carl\Desktop\Project\Maps\Webcam\Maps_new\testMap2.jpg"
    map = cv2.imread(map_path_offline)
    corner_mask_path = r"C:\Users\Carl\Desktop\Project\Mondamask.JPG"
    mask = cv2.imread(corner_mask_path)
    dimension_paper = [118.9,84.1] #cm A0
    dim = (int(dimension_paper[1]),int(dimension_paper[0]))
    # Switching red and blue channels
    map[:, :, [0, 2]] = map[:, :, [2, 0]]
    mask[:, :, [0, 2]] = mask[:, :, [2, 0]]


    p2_1, p98_1 = np.percentile(map, (2, 98))
    img_res1 = exposure.rescale_intensity(map, in_range=(p2_1,p98_1))
    img1_gray = cv2.cvtColor(img_res1, cv2.COLOR_BGR2GRAY)
    output = bg_clustering(img1_gray, (50,50))

    corner_location = corner_detection(output,mask) # Get the location of the 4 corners
    img_straighten, M = four_point_transform(map, corner_location) # Get the transformation matrix and the straighten img
    im_dim = img_straighten.shape
    obstacles = get_obstacles(img_straighten) # OFFLINE
    start = timeit.default_timer()
    thymio_coord = get_thymio_info(map,M,dim,im_dim) # Do these online, and feed info to kalman filter
    endpoint_coord = get_endpoint_info(map,M,dim,im_dim)
    low_res_img = cv2.resize(img_straighten, dsize=((dim[1], dim[0])))

    if save:
        plt.imsave('obstacles.jpg',output)
    stop = timeit.default_timer()
    #print('Time: ', stop - start)



def bg_clustering(image, px_zero,threshold_bg):
    list_p = [] # Place holder
    output = np.zeros_like(image) # place holder output img
    list_p.append((px_zero[0], px_zero[1])) # Get our initial background pixel picked
    while len(list_p):
        if len(list_p)<1: # sanity check to have a starting point
            break
        current_px = list_p[0] # Get the first pixel
        output[current_px[0], current_px[1]] = 255 # make it 255
        for coord in get_8_neighbors(current_px[0], current_px[1], image.shape): # Get the 8 neighbors of this pixel
            if abs((int(image[coord[0], coord[1]])))>threshold_bg and output[coord[0], coord[1]]<255: # If each of this neighbor is above a threshold, then its a background pixel
                output[coord[0], coord[1]] = 255 # Convert it to a 255 pixel
                list_p.append((coord[0], coord[1])) # append it to the list of background pixels
        list_p.pop(0) # Remove the initial pixel guess (in case we picked a wrong pixel). If its background, it will be picked later anyway
    return output

def get_8_neighbors(y, x, shape):
    out = [] # Matrix that will have the 8 neighbors
    # Get the 8 neighbors, unless its out of the picture borders
    if y-1 > 0 and x-1 > 0:
        out.append( (y-1, x-1))
    if y-1 > 0 :
        out.append( (y-1, x))
    if y-1 > 0 and x+1 < shape[1]:
        out.append( (y-1, x+1))
    if x-1 > 0:
        out.append( (y, x-1))
    if x+1 < shape[1]:
        out.append( (y, x+1))
    if y+1 < shape[0] and x-1 > 0:
        out.append( ( y+1, x-1))
    if y+1 < shape[0] :
        out.append( (y+1, x))
    if y+1 < shape[0] and x+1 < shape[1]:
        out.append( (y+1, x+1))
    return out

def get_obstacles(img):
    obstacles = black_contours(img,0.001) # Get the obstacles map
    fat_obstacles = process_obstacles(obstacles) # Clean the image and increase the size of the obstacles
    return fat_obstacles

def get_thymio_info(img,M,im_dim,dim):
    kernel = np.ones((5,5),np.float32)/25 # Get a kernel to filter
    img = cv2.filter2D(img,-1,kernel) # Smooth (blur) the image to reduce noise
    obstacles = get_obstacles(img) # Get the obstacles on the tilted image
    img[obstacles == [255]] = [0,0,0] # Remove the obstacles from our current image
    thymio_map = color_filtering(img,"blue") # Apply a blue filter to keep the dots on the thymio
    thymio_coords = end_point_start_point(thymio_map, 0.001, "thymio") # Get the thymio position (x,y of the centers of the two blue circles on the thymio)
    if (thymio_coords[0][0] > 0):       # if the thymio is not properly detected
        bigpt = thymio_coords[0] # Coords of the bigger circle on the thymio
        
        smallpt = thymio_coords[1] # Coords of the smaller circle on the thymio
        
        thymio_coord_big = tranformation_matrix(bigpt,M) # Get the location of the big circle in the straight image
        thymio_coord_small = tranformation_matrix(smallpt,M) # Get the location of the small circle in the straight image
        thymio_center_coord, thymio_orientation = orientation_location_thymio(thymio_coord_big, thymio_coord_small) # Get the orientation of the thymio (angle)
        thymio_center_coord = transformation_downgrade_coords(thymio_center_coord,im_dim,dim) # Get the coordinates in the small resolution image
        thymio_coord = [thymio_center_coord, thymio_orientation] # Concatinate the data
        #print("Thymio Coordinates + Orientation: ", thymio_coord)
        return thymio_coord
    else:
        thymio_coords = [(-1,-1), float("nan")]
        #print("Thymio not detected by Vision")
        return thymio_coords

def get_endpoint_info(img,M,im_dim,dim):
    
    endpoint_map = color_filtering(img,"green") # Apply a blue filter to keep the dots on the thymio
    endpoint_coord = end_point_start_point(endpoint_map, 0.001, "endpoint") # Get the endpoint position (x,y of the center of the star)
    if (endpoint_coord[0] > 0):
        endpoint_coord = tranformation_matrix(endpoint_coord,M) # Get the endpoint in the straight image
        endpoint_coord = transformation_downgrade_coords(endpoint_coord,im_dim,dim) # Get the coordinates in the small resolution image
        print("Endpoint Coordinates: ", endpoint_coord)
        return endpoint_coord
    else:
        endpoint_coord = [(-1,-1)]
        print("Endpoint not detected by Vision")
        return thymio_coords

def tranformation_matrix(pt,M):
    A = M[0:2,0:2]; # Rotation Matrix
    b = M[0:2,2]; # Translation Matrix
    tranformed_pt = np.matmul(A,pt) + b # Affine Tranformation from tilted image to straight img
    return tranformed_pt

def transformation_downgrade_coords(pt,im_dim,dim):
    x_new = pt[0]*im_dim[0]/dim[0] # convert x from the higher resolution to the lower resolution images
    y_new = pt[1]*im_dim[1]/dim[1] # convert y from the higher resolution to the lower resolution images
    return x_new,y_new

def process_obstacles(img):
    output = np.copy(img)
    kernel = np.ones((3,3), np.uint8)
    output = cv2.erode(output, kernel, iterations=1) # Erode and delate to remove isolated pixels and close the shapes
    output = cv2.dilate(output, kernel, iterations=2)
    kernel2 = np.ones((9,9), np.uint8)
    output = cv2.dilate(output, kernel2, iterations=2) # Increase to size of the obstacles to account for the size of the thymio in the path planning
    if Plot:
        plt.imshow(output)
        plt.show()
    return output

def order_points(pts):
    four_points = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    four_points[0] = pts[np.argmin(s)]
    four_points[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    four_points[1] = pts[np.argmin(diff)]
    four_points[3] = pts[np.argmax(diff)]
    return four_points

def four_point_transform(img, pts):
    four_points = order_points(pts) # Just in case the corners are not in the correct order
    (top_left, top_right, bottom_right, bottom_left) = four_points # Get each corner

    width_low = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2)) # Get the width of the lower part of the paper
    width_high = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2)) # Get the width of the upper part of the paper
    height_right = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + ((top_right[1] - bottom_right[1]) ** 2)) # Get the height of the left part of the paper
    height_left = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2)) # Get the height of the right part of the paper

    Width = max(int(width_low), int(width_high))
    Height = max(int(height_right), int(height_left))

    dimention_p = np.array([[0, 0],[Width - 1, 0],[Width - 1, Height - 1],[0, Height - 1]], dtype = "float32") # Get the location/dimension of the projection
    M = cv2.getPerspectiveTransform(four_points, dimention_p) # Get the transformation matrix
    img_straighten = cv2.warpPerspective(img, M, (Width, Height)) # Get the straighten image
    if Plot:
        plt.imshow(img_straighten)
        plt.show()
    return img_straighten, M

def corner_detection(img,mask):
    large_image = np.copy(img)
    small_image = np.copy(mask)
    method = cv2.TM_SQDIFF_NORMED # Method used for matching the template
    corner_location = np.zeros((4,2)) # Place holder for the location of the corners
    small_image = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY) # Convert to grayscale
    _, large_image = cv2.threshold(large_image, 30, 255, cv2.THRESH_BINARY) # Get binary image
    _, small_image = cv2.threshold(small_image, 30, 255, cv2.THRESH_BINARY) # Get binary image
    for i in range (0,4): # Do this 4 times, one time for each corner
        result = cv2.matchTemplate(small_image, large_image, method) # Find the corner in the image
        mn,_,mnLoc,_ = cv2.minMaxLoc(result) # Get the best match out of the results
        MPx,MPy = mnLoc # Extract the coordinates of the best match
        trows,tcols = small_image.shape[:2] # Get the size of the mask
        cv2.rectangle(large_image, (MPx,MPy),(MPx+tcols,MPy+trows),(0,0,255),2) # Draw the rectangle on large_image
        new_img = 255*np.ones(large_image.shape,np.uint8) # Place holder for the image this is only used to draw the mask on the image for debugging
        large_image[MPy:MPy+trows,MPx:MPx+trows] = new_img[MPy:MPy+trows,MPx:MPx+trows] # Draw the mask on the image
        small_image = cv2.rotate(small_image, cv2.ROTATE_90_CLOCKWISE) # Ritate the mask 90 degrees to match the next corner
        if i == 0: # These if conditions are to account for the rotation of the rectangle (not square) mask and get accurate coordinates
            corner_location[i,:] = [MPy,MPx]
        elif i == 1:
            corner_location[i,:] = [MPy,MPx+tcols]
        elif i == 2:
            corner_location[i,:] = [MPy+trows,MPx+tcols]
        else:
            corner_location[i,:] = [MPy+trows,MPx]
    corner_location = np.fliplr(corner_location) # Flip the array to get the requested shape

    return corner_location

def color_filtering(img,color):
    large_image = np.copy(img)
    if color == "blue":
        lower = np.array([20,50,90])    # lower color threshold
        upper = np.array([100,100,255]) # upper color threshold
    elif color == "green":
        lower = np.array([0,90,0]) # lower color threhsold
        upper = np.array([90,255,130]) # upper color threshold
    color_mask = cv2.inRange(large_image, lower, upper) # Create the mask with lower and upper threshold of RGB values
    large_image = cv2.bitwise_and(large_image, large_image, mask=color_mask) # Bitwise and to filter the pixels that are not of the desired color
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
    largest_areas = largest_areas[:-1] # remove the border of the paper (biggest area)
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
    _, threshold = cv2.threshold(output_grey, 30, 255, cv2.THRESH_BINARY) # threshold and obtain 0 and 255 values only
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
        try:
            M = cv2.moments(c) # calculating moments for each contour, i.e center of the circle englobing the contours
            cX = int(M["m10"] / M["m00"]) # calculate x coordinate of center
            cY = int(M["m01"] / M["m00"]) # calculate y coordinate of center
            cv2.circle(location_image, (cX, cY), 5, (255, 255, 255), -1) # Draw the circle englobing the contours
        except ZeroDivisionError as err:
            coordinates = [[-1,-1],[-1,-1]]
            break
            
        if (point == "thymio"):
            coordinates[i] = [cX, cY] # Assign coordinates
        else:
            coordinates = [cX, cY] # Assign coordinates
        i = i + 1
    if Plot:
        plt.imshow(location_image)
        plt.show()
    if(point == "thymio"): # if we are getting garbage as location of the thymio
        if (abs(coordinates[0][0]-coordinates[1][0]) > 40 or abs(coordinates[0][0]-coordinates[1][0]) > 40):
            coordinates = [[-1,-1],[-1,-1]]
    return coordinates

def orientation_location_thymio(bigpt, smallpt):

    center_thymio = [(bigpt[0]+smallpt[0])/2, (bigpt[1] + smallpt[1])/2] # Getting x and y of image (points are already sorted)
    slope = (bigpt[1]-smallpt[1])/(-bigpt[0]+smallpt[0]) # Obtain the slope of the thymio
    angle = math.degrees(math.atan(slope)) # Get the angle in degrees of the thymio
    if bigpt[0]>smallpt[0]:
        if bigpt[1]<smallpt[1]:
            angle = angle - 180
        else:
            angle = angle + 180
    return center_thymio, angle


if __name__ == "__main__":
    initialization()