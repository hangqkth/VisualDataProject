import matplotlib.pyplot as plt
import numpy as np
import cv2
from functionals import read_img, draw_kp, show_img
import scipy.signal
import os


data_img = read_img('./data1/obj1_5.JPG')
query_img = read_img('./data1/obj1_t1.JPG')

# show_img(data_img, 'database')
# show_img(query_img, 'query')

def sift_kp_extract(img):
    sift = cv2.xfeatures2d.SIFT_create(edgeThreshold=2.9, contrastThreshold=0.16, nOctaveLayers=5)
    kp_sift = sift.detect(img, None)
    return kp_sift

def surf_kp_extract(img):
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=8000)
    kp_surf, des = surf.detectAndCompute(data_img, None)
    return kp_surf


def crop_img(img):
    h, w = img.shape[0], img.shape[1]
    h_list, w_list = [], []
    for i in range(h):
        row_vector = img[i, :]
        if np.sum(row_vector) != 0:
            h_list.append(i)
    for j in range(w):
        column_vector = img[:, j]
        if np.sum(column_vector) != 0:
            w_list.append(j)
    return h_list[0], h_list[-1], w_list[0], w_list[-1]


def rotate_img(image_torotate, angle, gray=False):
    height, width = image_torotate.shape[:2]
    new_width = int(np.ceil(width * np.abs(np.cos(np.radians(angle))) + height * np.abs(np.sin(np.radians(angle)))))
    new_height = int(np.ceil(width * np.abs(np.sin(np.radians(angle))) + height * np.abs(np.cos(np.radians(angle)))))
    new_width_1 = new_width*2
    new_height_1 = new_height*2
    if gray:
        background = np.full((new_height_1, new_width_1, 1), 0, dtype=np.uint8)
    else:
        background = np.full((new_height_1, new_width_1, 3), (0, 0, 0), dtype=np.uint8)
    x_offset = (new_width_1 - width) // 2
    y_offset = (new_height_1 - height) // 2
    background[y_offset:y_offset+height, x_offset:x_offset+width] = image_torotate
    image_torotate = background

    center_x = new_width_1 // 2
    center_y = new_height_1 // 2
    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    rotated_image = cv2.warpAffine(image_torotate, rotation_matrix, (new_width_1, new_height_1), flags=cv2.INTER_LINEAR)

    x_start = (new_width_1 - new_width) // 2
    y_start = (new_height_1 - new_height) // 2

    x_end = x_start + new_width
    y_end = y_start + new_height

    rotated_image = rotated_image[y_start:y_end, x_start:x_end]
    return rotated_image


def generate_point_image(img, kp_list):
    kp_img = np.zeros((img.shape[0], img.shape[1]))
    for p in range(len(kp_list)):
        coordinate = np.floor(kp_list[p].pt)
        kp_img[int(coordinate[1]), int(coordinate[0])] = 1
    return kp_img


def get_point_location(kp_img):
    none_zero_idx = np.nonzero(kp_img)
    coordinate_list = [[none_zero_idx[1][i], none_zero_idx[0][i]] for i in range(len(none_zero_idx[0]))]
    return coordinate_list


def sift_find_match(source_img, angle):
    kp_sift = sift_kp_extract(source_img)
    kp_img = generate_point_image(source_img, kp_sift)

    image_rotated = rotate_img(data_img, angle)
    kp_rotated = rotate_img(np.expand_dims(kp_img, 2), angle, gray=True)

    kp_origin_rotated = get_point_location(kp_rotated)
    kp_rotated = sift_kp_extract(image_rotated)
    draw_kp(kp_rotated, image_rotated)
    draw_kp(kp_origin_rotated, image_rotated, cv_point=False)

    match_num = 0
    for i in range(len(kp_origin_rotated)):
        p = np.array(kp_origin_rotated[i])
        for j in range(len(kp_rotated)):
            target = np.floor(np.array(list(kp_rotated[j].pt)))
            distance = np.sum(np.abs(p-target))
            match_num += 1 if distance < 2 else 0
    return match_num, match_num/len(kp_origin_rotated)


def test_robustness(img, angle_list):
    rate_list = []
    for angle in angle_list:
        matches, robust_rate = sift_find_match(img, angle)
        rate_list.append(robust_rate)
    plt.plot(angle_list, rate_list)
    plt.show()


def imresize(image, target_size):
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

    return resized_image

def find_match_scale(img, scale, type):
    weight =np.floor(img.shape[0]*scale)
    height =np.floor(img.shape[1]*scale)
    print(weight, height)


    scaled_img=imresize(img, (int(height), int(weight)))

    #cv2.imwrite('output.jpg', img)
    #cv2.imwrite('output_scaled.jpg', scaled_img)
    #exit(2)


    if type=="sift":
        kp_sift = sift_kp_extract(img)
        kp_sift_scaled = sift_kp_extract(scaled_img)

    if type=="surf":
        kp_sift = surf_kp_extract(img)
        kp_sift_scaled = surf_kp_extract(scaled_img)


    match_num = 0
    for i in range(len(kp_sift)):
        p = np.array([kp_sift[i].pt[0], kp_sift[i].pt[1]])
        count = 0
        for j in range(len(kp_sift_scaled)):
            target = np.floor(np.array(list(kp_sift_scaled[j].pt)))
            distance = np.sum(np.abs(p - target))
            count += 1 if distance < 2 else 0
        match_num += 1 if count >0 else 0
    return match_num, match_num / len(kp_sift)


def test_scaling_factor(img, n, type):
    rate_list_sift = []
    for i in range(n):
        matches_sift, x =find_match_scale(img, 1.2**i,type)
        rate_list_sift.append(x)
    scale_list=np.arange(0,n,1)
    plt.plot(scale_list,rate_list_sift)
    plt.show()


angles = np.arange(0, 360, 15)

#test_robustness(data_img, angles)

test_scaling_factor(data_img, 3, "sift")

#test_scaling_factor(data_img, 8, "surf")

print(data_img.size)