import numpy as np
from matplotlib import pyplot as plt
import scipy.ndimage as ndimage
import cv2
import heapq
from sklearn.cluster import KMeans
import timeit

offline = True
fast = True
Plot = False
save = False

def initialization():
    start = timeit.default_timer()
    map_path_offline = r"C:\Users\Carl\Desktop\Project\map3.JPG"
    map = cv2.imread(map_path_offline)
    corner_mask_path = r"C:\Users\Carl\Desktop\Project\mask_corner.JPG"
    mask = cv2.imread(corner_mask_path)
    dimension_paper = [841,1189] #mm
    # Switching red and blue channels
    map[:, :, [0, 2]] = map[:, :, [2, 0]]
    mask[:, :, [0, 2]] = mask[:, :, [2, 0]]

    corner_location = corner_detection(map,mask)
    img_straighten = four_point_transform(map, corner_location)
    img_straighten_grey = cv2.cvtColor(img_straighten, cv2.COLOR_RGB2GRAY)
    if fast:
        obstacles = color_filtering(img_straighten,"black") # OR  obstacles = find_contours(img_straighten_grey, 0.001)
        endpoint = color_filtering(img_straighten,"purple")
    else:
        obstacles, end_point = clustering(img_straighten)
    if save:
        plt.imsave('obstacles.jpg',obstacles)
    thymio = color_filtering(img_straighten,"green and blue")
    coordinates_thymio, center_thymio, slope_thymio, direction_thymio = orientation_location_thymio(thymio)
    modified_obstacles = process_obstacles(obstacles)
    stop = timeit.default_timer()
    if Plot:
        plt.imshow(obstacles)
        plt.show()
    print('Time: ', stop - start)

def process_obstacles(img):
    output = np.copy(img)
    kernel = np.ones((3,3), np.uint8)
    output = cv2.erode(output, kernel, iterations=1)
    output_ = cv2.dilate(output, kernel, iterations=2)
    return output

def preprocessing(img):
    pass


def clustering(img):
    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = img.reshape((-1, 3))
    # convert to float
    pixel_values = np.float32(pixel_values)
    # define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    k = 7
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(img.shape)
    thymio_blue = np.copy(img)
    thymio_green = np.copy(img)
    obstacles = np.copy(img)
    end_point = np.copy(img)
    labels = labels.reshape(img.shape[0],img.shape[1])
    (values,counts) = np.unique(labels,return_counts=True)
    background_i,obstacles_i,others_i,endpoint_i = heapq.nlargest(4, range(len(counts)), key=counts.__getitem__)

    obstacles[labels != obstacles_i] = [255, 255, 255]
    end_point[labels != endpoint_i] = [255, 255, 255]

    return obstacles, end_point


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

    # return the warped image
    return warped



def corner_detection(img,mask):
    large_image = np.copy(img)
    small_image = np.copy(mask)
    method = cv2.TM_SQDIFF_NORMED
    large_image = color_filtering(large_image,"red")
    if Plot:
        plt.imshow(large_image)
        plt.show()
    corner_location = np.zeros((4,2))
    for i in range (0,4):
        result = cv2.matchTemplate(small_image, large_image, method)
        mn,_,mnLoc,_ = cv2.minMaxLoc(result) # We want the minimum squared difference
        MPx,MPy = mnLoc # Extract the coordinates of our best match
        trows,tcols = small_image.shape[:2] # Get the size of the template. This is the same size as the match.
        # cv2.rectangle(large_image, (MPx,MPy),(MPx+tcols,MPy+trows),(0,0,255),2) # Draw the rectangle on large_image
        new_img = 255*np.ones(large_image.shape,np.uint8)
        large_image[MPy:MPy+trows,MPx:MPx+trows] = new_img[MPy:MPy+trows,MPx:MPx+trows]
        small_image = cv2.rotate(small_image, cv2.ROTATE_90_CLOCKWISE)
        # Changes to account for the rotation of the mask
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
        lower = np.array([90,0,0])
        upper = np.array([255,50,50])
    elif color == "blue":
        lower = np.array([0,0,90])
        upper = np.array([50,50,255])
    elif color == "green":
        lower = np.array([0,90,0])
        upper = np.array([50,255,50])
    elif color == "green and blue":
        lower_b = np.array([0,0,90])
        upper_b = np.array([50,50,255])
        lower_g = np.array([0,90,0])
        upper_g = np.array([50,255,50])
        color_mask_b = cv2.inRange(large_image, lower_b, upper_b)
        color_mask_g = cv2.inRange(large_image, lower_g, upper_g)
        large_image_b = cv2.bitwise_and(large_image, large_image, mask=color_mask_b)
        large_image_g = cv2.bitwise_and(large_image, large_image, mask=color_mask_g)
        large_image = large_image_b + large_image_g
        return large_image
    elif color == "purple":
        lower = np.array([50,0,50])
        upper = np.array([255,30,255])
    elif color == "black":
        large_image = cv2.cvtColor(large_image, cv2.COLOR_RGB2GRAY)
        _,output = cv2.threshold(large_image,20,255,cv2.THRESH_BINARY)
        return output
    color_mask = cv2.inRange(large_image, lower, upper)
    large_image = cv2.bitwise_and(large_image, large_image, mask=color_mask)
    return large_image

def find_contours(img,constant):
    _, threshold = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)
    contours,_ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_contours = np.zeros([img.shape[0],img.shape[1]])
    largest_areas = sorted(contours, key=cv2.contourArea)
    largest_areas = largest_areas[:-1] # remove sorted and [:-1] to get the corners and the outer borders too
    largest_areas = largest_areas[::-1]
    for cnt in largest_areas[:3]: # TODO: Find a thresholding way to remove everything and only keep the obstacles
        approx = cv2.approxPolyDP(cnt, constant*cv2.arcLength(cnt, True), True)
        cv2.drawContours(img_contours, [approx], 0, (255), thickness=cv2.FILLED) #  replace thickness=cv2.FILLED with 5 for edges only
        x = approx.ravel()[0]
        y = approx.ravel()[1]
    if Plot:
        plt.imshow(img_contours)
        plt.show()
    return img_contours

def orientation_location_thymio(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, threshold = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    contours,_ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    coordinates_thymio = np.zeros((2,2))
    direction_thymio = 0
    i = 0
    for c in contours:
        # calculate moments for each contour
        M = cv2.moments(c)
        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        coordinates_thymio[i] = [cX, cY]
        cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)
        i = i + 1
    if Plot:
        plt.imshow(img)
        plt.show()
    if (coordinates_thymio[0][0]-coordinates_thymio[1][0] == 0):
        slope_thymio = coordinates_thymio[0][1]
    else:
        slope_thymio = (coordinates_thymio[0][1] - coordinates_thymio[1][1])/(coordinates_thymio[0][0]-coordinates_thymio[1][0])
    center_thymio = [(coordinates_thymio[0][1] + coordinates_thymio[1][1])/2, (coordinates_thymio[0][0]+coordinates_thymio[1][0])/2]
    # TODO: recheck these for orientation
    if (coordinates_thymio[0][1] > coordinates_thymio[1][1]): # 1st and 4th quadrants are positive
        direction_thymio = 0
    elif (coordinates_thymio[0][1] == coordinates_thymio[1][1]):
        if(coordinates_thymio[0][0] > coordinates_thymio[1][0]):
            direction_thymio = 1
        else:
            direction_thymio = 0
    return coordinates_thymio, center_thymio, slope_thymio, direction_thymio


def sharpen(img):
    output = img.copy()
    # Sharpening mask
    D = np.array([[-1,-2,-1],
                  [-2,20,-2],
                  [-1,-2,-1]])
    output = cv2.filter2D(output,-1,D)
    # output = cv2.normalize(output,  norm_img, 0, 255, cv2.NORM_MINMAX)
    return output

def filter_isolated_cells(array, struct):
    filtered_array = np.copy(array)
    id_regions, num_ids = ndimage.label(filtered_array, structure=struct)
    id_sizes = np.array(ndimage.sum(array, id_regions, range(num_ids + 1)))
    area_mask = (id_sizes == 1)
    filtered_array[area_mask[id_regions]] = 0
    return filtered_array


if __name__ == "__main__":
    initialization()




# kernel = np.ones((5,5),np.uint8)

# pts_transposed = [[corner_location[j][i] for j in range(len(corner_location))] for i in range(len(corner_location[0]))]

# map[int(corner_location[0,0]),int(corner_location[0,1])] = [0,50,255]
# map[int(corner_location[1,0]),int(corner_location[1,1])] = [0,50,255]
# map[int(corner_location[2,0]),int(corner_location[2,1])] = [0,50,255]
# map[int(corner_location[3,0]),int(corner_location[3,1])] = [0,50,255]
