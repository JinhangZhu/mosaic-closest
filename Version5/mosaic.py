#H#######################################################################
# FILENAME :        mosaic.py             
#
# DESIGN REF :  https://stackoverflow.com/questions/34264710/what-is-the-point-of-floatinf-in-python
#               https://docs.python.org/2/library/multiprocessing.html#multiprocessing-programming
#               https://docs.opencv.org/3.1.0/d1/db7/tutorial_py_histogram_begins.html
#               https://github.com/pythonml/mosaic/blob/master/main.py
#               https://stackoverflow.com/questions/38598118/difference-between-plt-show-and-cv2-imshow
#               https://stackoverflow.com/questions/15393216/create-multidimensional-zeros-python
#               https://research.wmz.ninja/articles/2018/03/on-sharing-large-arrays-when-using-pythons-multiprocessing.html   
#               
#      
# DESCRIPTION :
#       Create a Mosaic of images that builds up an image.
#
# AUTHOR :          Jinhang <jinhang.d.zhu@gmail.com>
# VERSION :         1.5   
# START DATE :      17 Mar. 2019
#
#H#

import ctypes
import re
import os
import multiprocessing as mp
from multiprocessing.sharedctypes import RawArray
# from numpy import ctypeslib
import numpy as np 
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from tqdm import tqdm
import time

IMG_DIR = './images'

RATIO = 10

#
#   Get the luminance of an image
# #
def get_luminance(source):
    return np.sum(source)/np.size(source)
#   The luminance array of all luminances of images
def get_luminance_arr(resource_images):
    arr_luminance = np.zeros(len(resource_images))
    for i, resource_image in enumerate(resource_images):
        arr_luminance[i] = get_luminance(resource_image)
    return arr_luminance


#
#   Get the dynamic range of an image
# #
def get_dy_range(source):
    return np.amin(source), np.amax(source)

def get_dy_range_arr(resource_images):
    arr_dy_range = np.zeros(shape=(1,len(resource_images),2))
    for i, resource_image in enumerate(resource_images):
        arr_dy_range[:,i,:] = get_dy_range(resource_image)
    return arr_dy_range

#
#   Resize the image to be in the shape of (height, width) without distortion
# #
def resize(source, height, width):
    shape_row = source.shape[0]
    shape_col = source.shape[1]
    shrink_ratio = min(shape_row/height, shape_col/width)
    resized = cv2.resize(source, (int(shape_col/shrink_ratio)+1, int(shape_row/shrink_ratio)+1), interpolation=cv2.INTER_CUBIC)
    result = resized[:height, :width, :]
    return result

#
#   Calculate the euclidean distance between two images
# #
def img_distance(source1, source2):
    if source1.shape != source2.shape:
        msg = "shapes are different {} {}".format(source1.shape, source2.shape)
        raise Exception(msg)
    array_1 = source1.flatten()
    array_2 = source2.flatten()
    dist = euclidean(array_1, array_2)
    return dist

# 
#   Calculate the euclidean distance between two histograms
# #
def hist_distance(source1, source2):
    color = ('b', 'g', 'r')
    hist_1 = np.zeros((256, 1, 3))
    hist_2 = np.zeros((256, 1, 3))
    dist = np.zeros((1,3))
    for i, col in enumerate(color):
        hist_1[:, :, i] = cv2.calcHist([source1],[i],None,[256],[0,256])
        hist_2[:, :, i] = cv2.calcHist([source2],[i],None,[256],[0,256])
    array_1 = hist_1.flatten()
    array_2 = hist_2.flatten()
    dist = euclidean(array_1, array_2)
    return dist
# 
#   Calculate the euclidean distance between two histograms in channels
# #
def hist_distance_channel(source1, source2):
    color = ('b', 'g', 'r')
    hist_1 = np.zeros((256, 1, 3))
    hist_2 = np.zeros((256, 1, 3))
    for i, col in enumerate(color):
        hist_1[:, :, i] = cv2.calcHist([source1],[i],None,[256],[0,256])
        hist_2[:, :, i] = cv2.calcHist([source2],[i],None,[256],[0,256])
    dist_b = euclidean(hist_1[:, :, 0], hist_2[:, :, 0])
    dist_g = euclidean(hist_1[:, :, 1], hist_2[:, :, 1])
    dist_r = euclidean(hist_1[:, :, 2], hist_2[:, :, 2])
    return dist_b, dist_g, dist_r

#
#   Load images in a specific directory
# #
def load_images(height, width):
    img_dir = IMG_DIR
    filenames = os.listdir(img_dir)
    result = []
    print(len(filenames))
    for filename in tqdm(filenames):
        if not re.search('.jpg', filename, re.I):
            continue
        try:
            filepath = os.path.join(img_dir, filename)
            source_im = cv2.imread(filepath)
            height_im = source_im.shape[0]
            width_im = source_im.shape[1]
            if height != height_im or width != width_im:
                source_im = resize(source_im, height, width)
            result.append(np.array(source_im))
        except Exception as e:
            msg = 'error with {} - {}'.format(filepath, str(e))
            print(msg)
    return np.array(result, dtype = np.uint8)

# 
#   Find the similarist image from the resource images by comparing euclidean distance
# #
def find_closest_image(q, shared_resource_images, resource_images_shape, shared_result, img_shape, set_height, set_width):
    shared_images_array = np.frombuffer(shared_resource_images, dtype=np.uint8)
    resource_images = shared_images_array.reshape(resource_images_shape)
    #   Get luminance, dynamic range and histogram distance all in arrays
    #   in order to avoid iterating.
    # arr_lumi_img = np.frombuffer(shared_arr_lumi_img, dtype=np.float64)
    # arr_dy_ran_img = np.frombuffer(shared_arr_dy_ran_img, dtype=np.float64)

    while True:
        [row, col, pad] = q.get()
        print('row: {}, col: {}'.format(row, col))
        if len(pad.shape) is 3: #   3 channels
            # min_dist_img = float("inf")     #   It acts as an unbounded upper value for comparison. This is useful for finding lowest values for something
            # min_dist_hist = float("inf")
            min_dist_b = float("inf")
            min_dist_g = float("inf")
            min_dist_r = float("inf")
            min_diff_lumi_b = 255
            min_diff_lumi_g = 255
            min_diff_lumi_r = 255
            min_img = None
            
            for i, resource_image in enumerate(resource_images):
                # 
                # Calculate euclidean distance between the image and the pad
                # #
                # dist_img = img_distance(pad, resource_image)
                # dist_hist = hist_distance(pad, resource_image)
                dist_b, dist_g, dist_r = hist_distance_channel(pad, resource_image)
                # 1
                # Auxiliary methods to eliminate converse-looking images
                # #
                diff_lumi_b = abs(get_luminance(resource_image[:,:,0]) - get_luminance(pad[:,:,0]))
                diff_lumi_g = abs(get_luminance(resource_image[:,:,1]) - get_luminance(pad[:,:,1]))
                diff_lumi_r = abs(get_luminance(resource_image[:,:,2]) - get_luminance(pad[:,:,2]))
                # if abs(arr_lumi_img[i] - get_luminance(pad)) < 50:
                    # min_im, max_im = arr_dy_ran_img[0,i,0], arr_dy_ran_img[0,i,1]
                    # min_pad, max_pad = get_dy_range(pad)
                    # if abs(min_im - min_pad) < 50 or abs(max_im - max_pad) < 50: 
                        # if dist_img < min_dist_img and dist_hist < min_dist_hist:
                        #     min_dist_img = dist_img
                        #     min_dist_hist = dist_hist
                        #     min_img = resource_image
                state_hist = dist_b < min_dist_b and dist_g < min_dist_g and dist_r < min_dist_r
                state_lumi = diff_lumi_b < min_diff_lumi_b and diff_lumi_g < min_diff_lumi_g and diff_lumi_r < min_diff_lumi_r
                state_thres = diff_lumi_b < 50 and diff_lumi_g < 50 and diff_lumi_r < 50
                if state_thres:
                    if state_hist and state_lumi:
                        min_diff_lumi_b = diff_lumi_b
                        min_diff_lumi_g = diff_lumi_g
                        min_diff_lumi_r = diff_lumi_r

                        min_dist_b = dist_b
                        min_dist_g = dist_g
                        min_dist_r = dist_r

                        min_img = resource_image
                        # if dist_hist < min_dist_hist:
                        #     min_dist_hist = dist_hist
                        #     min_img = resource_image
                # Calculate euclidean distance between two histograms
                # # Calculate euclidean distance between two histograms in channels

            im_res = np.frombuffer(shared_result, dtype=np.uint8).reshape(img_shape)
            im_res[row:row+set_height, col:col+set_width, :] = min_img

        elif len(pad.shape) is 2:
            # min_dist_img = float("inf")     #   It acts as an unbounded upper value for comparison. This is useful for finding lowest values for something
            # min_dist_hist = float("inf")
            min_dist_hist = float("inf")
            min_diff_lumi = 255
            min_img = None
            
            for i, resource_image in enumerate(resource_images):
                
                # 
                # Calculate euclidean distance between the image and the pad
                # #
                # dist_img = img_distance(pad, resource_image)
                # dist_hist = hist_distance(pad, resource_image)
                dist_hist = hist_distance(pad, resource_image)
                # 1
                # Auxiliary methods to eliminate converse-looking images
                # #
                diff_lumi = abs(get_luminance(resource_image) - get_luminance(pad))
                state_hist = dist_hist < min_dist_hist
                state_lumi = diff_lumi < min_diff_lumi
                state_thres = diff_lumi < 50
                if state_thres:
                    if state_hist and state_lumi:
                        min_diff_lumi = diff_lumi
                        min_dist_hist = dist_hist
                        resource_image = cv2.cvtColor(resource_image, cv2.COLOR_BGR2GRAY)
                        min_img = resource_image

            im_res = np.frombuffer(shared_result, dtype=np.uint8).reshape(img_shape)
            im_res[row:row+set_height, col:col+set_width, :] = min_img
        # 
        # Necessary method of JoinableQueue
        # To terminate the finished process
        # #
        q.task_done()

#   Return a fixed shape according to the original shape
def get_set_shape(shape):
    return [32, 32]

#   Generate the mosaic with the resource image file and the output file indicated
def generate_mosaic(infile, outfile):
    print('Reading the background image: ' + infile)
    img = cv2.imread(infile)
    set_height, set_width = get_set_shape(img.shape)
    img_shape = list(img.shape)

    #   Make corresponding shape of the full image accroding to the set shape of a single one
    img_shape[0] = int(img_shape[0]/set_height) * set_height * RATIO
    img_shape[1] = int(img_shape[1]/set_width) * set_width * RATIO
    print('Resizing the background image...')
    img = cv2.resize(img, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_CUBIC)
    # 
    # #
    # REF:  cv2.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]]) → dst
    # dsize = Size(round(dst.cols), round(dst.rows))
    # #INTER_CUBIC - a bicubic interpolation over 4x4 pixel neighborhood
    #
    #   Print the shape of the output image
    print('Shape of the output image: {}'.format(img_shape))

    #   
    im_res = np.zeros(img_shape, np.uint8)
    #
    #   All resource image in the set shape to be used
    # #
    print('Loading images as patches...')
    resource_images = load_images(set_height, set_width)
    #
    #   Get the shape of the images
    # #
    resource_images_shape = resource_images.shape
    #
    #   Return a ctypes array allocated from shared memory
    #   The ctypes array is of the same size of what needs to be shared across multiple processes
    # #
    shared_resource_images = RawArray(ctypes.c_ubyte, len(resource_images.flatten()))
    # 
    # np.frombuffer:    Intepret shared_resource_images as a 1-dimensional array
    # np.coyto:         Copy the values from the array: shared_resource_images to another array: resource_images
    # #
    np.copyto(np.frombuffer(shared_resource_images, dtype=np.uint8).reshape(resource_images_shape), resource_images)
    # 
    # Reruen a ctypes array allocated from shared memory
    # The ctypes array is in the shape of the flattened output image "pool"
    # #  
    shared_result = RawArray(ctypes.c_ubyte, len(im_res.flatten()))

    # Luminance array and dynamic range of resource images
    arr_lumi_img = get_luminance_arr(resource_images)
    # arr_lumi_img_shape = arr_lumi_img.shape
    # shared_arr_lumi_img = RawArray('d', len(arr_lumi_img.flatten()))
    # np.copyto(np.frombuffer(shared_arr_lumi_img, dtype=np.float64).reshape(arr_lumi_img_shape), arr_lumi_img)

    # arr_dy_ran_img = get_dy_range_arr(resource_images)
    # arr_dy_ran_img_shape = arr_dy_ran_img.shape
    # shared_arr_dy_ran_img = RawArray('d', len(arr_dy_ran_img.flatten()))
    # np.copyto(np.frombuffer(shared_arr_dy_ran_img, dtype=np.float64).reshape(arr_dy_ran_img_shape), arr_dy_ran_img)
    # 
    # Create a Queue subclass, a queue which additionally has task_done() and join() methods.
    # #
    join_queue = mp.JoinableQueue()
    for i in range(5):
        p = mp.Process(target=find_closest_image, # The callable object to be invoked
            name='Process: {}'.format(i),
            args=(join_queue, shared_resource_images, resource_images_shape, shared_result, img_shape, set_height, set_width),
            daemon=True) #  Make daemon process finish execution
        p.start()
        print('Started process {}'.format(i+1))
    
    #
    #   Generate the pads through multiple processes
    # #
    for row in range(0, img_shape[0], set_height):
        for col in range(0, img_shape[1], set_width):
            pad = img[row:row+set_height, col:col+set_width, :]
            #
            #   Put the value in the queue: Which firstly finishes calculation firstly processes the value
            # #
            join_queue.put([row, col, pad])

    join_queue.join()
    # 
    # Output image file
    # Multiple processes all modify the file in order
    # #
    print('Writing the output image: {}'.format(outfile))
    cv2.imwrite(outfile, np.frombuffer(shared_result, dtype=np.uint8).reshape(img_shape))
    print('Happy ending.')

if __name__ == "__main__":
    str_infile = "mercy_down.jpg"
    str_outfile = "BGR_CHANNEL_NOTHRES_466.jpg"
    t = time.time()
    generate_mosaic(str_infile, str_outfile)
    elapesd = time.time() - t
    print('Duration time: {}'.format(elapesd))
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(cv2.imread(str_infile), cv2.COLOR_BGR2RGB)) # OpenCV stores images in BGR order instead of RGB.
    plt.title(str_infile)
    plt.subplot(1,2,2)
    plt.imshow(cv2.cvtColor(cv2.imread(str_outfile), cv2.COLOR_BGR2RGB))
    plt.title(str_outfile)
    plt.show()