import matplotlib.pyplot as plt
import numpy as np
import cv2
from functionals import read_img, draw_kp, show_img
from tqdm import tqdm
import scipy.signal
import os


def sift_kp_extract(img):
    sift = cv2.xfeatures2d.SIFT_create(edgeThreshold=2.9, contrastThreshold=0.16, nOctaveLayers=5, nfeatures=400)
    kp_sift = sift.detect(img, None)
    return kp_sift


def surf_kp_extract(img):
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=8000)
    kp_surf = surf.detect(img, None)
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


def find_match_rotate(source_img, angle, method):
    kp = sift_kp_extract(source_img) if method == "sift" else surf_kp_extract(source_img)
    kp_img = generate_point_image(source_img, kp)

    image_rotated = rotate_img(data_img, angle)
    kp_rotated = rotate_img(np.expand_dims(kp_img, 2), angle, gray=True)

    kp_origin_rotated = get_point_location(kp_rotated)
    kp_rotated = sift_kp_extract(image_rotated) if method == "sift" else surf_kp_extract(image_rotated)
    draw_kp(kp_origin_rotated, image_rotated, cv_point=False, title="origin_rotated")
    draw_kp(kp_rotated, image_rotated, title="kp_rotated")
    match_num = 0
    for i in tqdm(range(len(kp_origin_rotated))):
        p = np.array(kp_origin_rotated[i])
        distances = []
        for j in range(len(kp_rotated)):
            target = np.floor(np.array(list(kp_rotated[j].pt)))
            distances.append(np.max(np.abs(p - target)))
        match_num += 1 if min(distances) < 2 else 0
    return match_num, match_num/len(kp_origin_rotated)


def test_robustness(img, angle_list, method):
    rate_list = []
    for angle in angle_list:
        matches, robust_rate = find_match_rotate(img, angle, method)
        rate_list.append(robust_rate)
    plt.plot(angle_list, rate_list)
    plt.show()
    return rate_list


def im_resize(image, scale):
    weight = int(np.floor(image.shape[0] * scale))
    height = int(np.floor(image.shape[1] * scale))
    resized_image = cv2.resize(image, (height, weight), interpolation=cv2.INTER_LINEAR)
    return resized_image


def find_match_scale(img, scale, type):

    scaled_img = im_resize(img, scale)

    if type == "sift":
        kp = sift_kp_extract(img)
        kp_scaled = sift_kp_extract(scaled_img)

    elif type == "surf":
        kp = surf_kp_extract(img)
        kp_scaled = surf_kp_extract(scaled_img)

    else:
        kp, kp_scaled = None, None

    print("key points extract finished")

    kp_origin_scaled = [np.array(list(kp[i].pt)) * scale for i in range(len(kp))]
    # draw_kp(kp_origin_scaled, scaled_img, cv_point=False, title='origin scaled, ' + str(scale))
    # draw_kp(kp_scaled, scaled_img, title='scaled, '+str(scale))

    match_num = 0

    for i in tqdm(range(len(kp_origin_scaled))):
        p = np.array(kp_origin_scaled[i])
        distances = []
        for j in range(len(kp_scaled)):
            target = np.floor(np.array(list(kp_scaled[j].pt)))
            # print(p.shape, target.shape)
            distances.append(np.max(np.abs(p - target)))
        match_num += 1 if min(distances) < 2 else 0
    print("match finished")
    return match_num, match_num / len(kp_origin_scaled)


def test_scaling_factor(img, n, type):
    rate_list_sift = []
    for i in range(n):
        print("current scale: ", i)
        matches_sift, x = find_match_scale(img, 1.2**i, type)
        rate_list_sift.append(x)
    scale_list = np.arange(0, n, 1)
    plt.plot(scale_list, rate_list_sift)
    plt.show()
    return rate_list_sift


def draw_scale_plot():
    sift_scale_rate = np.load('./data1/scale_rate_sift.npy')
    surf_scale_rate = np.load('./data1/scale_rate_surf.npy')
    scale_list = [1.2*i for i in np.arange(0, 8, 1)]
    plt.plot(scale_list, sift_scale_rate)
    plt.plot(scale_list, surf_scale_rate)
    plt.legend(["sift", "surf"])
    plt.title("Repeatability versus scaling factor")
    # plt.show()


def draw_rotate_plot():
    sift_rotate_rate = np.load('./data1/rotate_rate_sift.npy')
    surf_rotate_rate = np.load('./data1/rotate_rate_surf.npy')
    angles = np.arange(0, 360, 15)
    plt.plot(angles, sift_rotate_rate)
    plt.plot(angles, surf_rotate_rate)
    plt.legend(["sift", "surf"])
    plt.title("Repeatability versus rotation angle")
    # plt.show()


if __name__ == "__main__":
    data_img = read_img('./data1/obj1_5.JPG')
    query_img = read_img('./data1/obj1_t1.JPG')
    angles = np.arange(0, 360, 15)
    method = "surf"

    # repeatability versus scaling factor
    # rate_list = test_scaling_factor(data_img, 8, method)
    # np.save('./data1/scale_rate_'+method+'.npy', np.array(rate_list))

    # repeatability versus rotation angle
    # rate_list = test_robustness(data_img, angles, method)
    # np.save('./data1/rotate_rate_'+method+'.npy', np.array(rate_list))
    # plt.figure(figsize=(10, 8))
    # plt.subplot(2, 1, 1)
    # draw_scale_plot()
    # plt.subplot(2, 1, 2)
    # draw_rotate_plot()
    # plt.show()
    kp_sift1 = sift_kp_extract(data_img)
    kp_sift2 = surf_kp_extract(query_img)

    def plot_subplot(img_array, kp_list, title, cv_point=True):

        plt.imshow(np.stack((img_array[:, :, 2], img_array[:, :, 1], img_array[:, :, 0]), axis=-1))
        for k in range(len(kp_list)):
            coordinate = kp_list[k].pt if cv_point else kp_list[k]
            plt.scatter(coordinate[0], coordinate[1], c='b', s=5, marker='*')
        plt.title(title)


    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plot_subplot(data_img, kp_sift1, "SIFT on obj1_5.JPG")
    plt.subplot(1, 2, 2)
    plot_subplot(query_img, kp_sift2, "SIFT on obj1_t5.JPG")
    plt.show()
