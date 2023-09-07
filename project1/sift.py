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

sift = cv2.xfeatures2d.SIFT_create(edgeThreshold=2.9, contrastThreshold=0.16, nOctaveLayers=5, sigma=1.4)
kp_sift = sift.detect(data_img, None)
kp_num = len(kp_sift)
print('key points number = ', kp_num)


draw_kp(kp_sift, data_img)


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






kp_img = generate_point_image(data_img, kp_sift)

image_rotated = rotate_img(data_img, 30)
kp_rotated = rotate_img(np.expand_dims(kp_img, 2), 30, gray=True)
kp_list_rotated = get_point_location(kp_rotated)

plt.imshow(np.stack((image_rotated[:, :, 2], image_rotated[:, :, 1], image_rotated[:, :, 0]), axis=-1))
for k in range(len(kp_list_rotated)):
    coordinate = kp_list_rotated[k]
    plt.scatter(coordinate[0], coordinate[1], c='b', s=5, marker='*')
plt.show()


