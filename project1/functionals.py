import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
import os


def read_img(img_path):
    img = cv2.imread(img_path)
    return img


def show_img(img_array, img_name=None):
    plt.figure(figsize=(10, 10))
    if len(img_array.shape) == 3:
        plt.imshow(np.stack((img_array[:, :, 2], img_array[:, :, 1], img_array[:, :, 0]), axis=-1))
        plt.title(img_name)
        plt.show()
    else:
        plt.imshow(img_array)
        plt.title(img_name)
        plt.show()
    # cv2.imshow('0', img_array)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def normalize(matrix):
    return (matrix-np.min(matrix))/(np.max(matrix) - np.min(matrix))



def sv_gaussian(scale, x, y):
    sigma = scale  # standard deviation
    g = (1/(2*np.pi*sigma**2))*np.exp(-(x**2+y**2)/(2*sigma**2))  # Gaussian pdf
    return g


def get_gaussian_img(imsize, scale):
    """return a Gaussian image for a given image size"""
    height, length = imsize[0], imsize[1]
    x_range = range(-length // 2, length // 2)
    y_range = range(-height // 2, height // 2)
    g_2d = np.zeros(imsize)
    for i in range(len(x_range)):
        for j in range(len(y_range)):
            g_2d[j, i] = sv_gaussian(scale, x_range[i], y_range[j])
    return g_2d


def load_gaussian_img(obj_img, k):
    available_kernel = os.listdir('./data1')
    kernel_name = 'gauss_'+str(k)+'.npy'
    if kernel_name in available_kernel:
        kernel = np.load('./data1/'+kernel_name)
    else:
        kernel = get_gaussian_img([obj_img.shape[0], obj_img.shape[1]], k)
        np.save('./data1/gauss_' + str(k) + '.npy', kernel)
    return kernel


def conv_for_rgb_img(kernel, img):
    """for rgh image, implement 2d conv on each channel dimension separately"""
    conv_out = []
    for i in range(img.shape[-1]):
        conv_out.append(normalize(scipy.signal.convolve(kernel, img[:, :, i], mode='same')))
    output = np.stack(conv_out, axis=-1)
    return output


def draw_kp(kp_list, img_array, title, cv_point=True):
    plt.figure(10)
    plt.imshow(np.stack((img_array[:, :, 2], img_array[:, :, 1], img_array[:, :, 0]), axis=-1))
    for k in range(len(kp_list)):
        coordinate = kp_list[k].pt if cv_point else kp_list[k]
        plt.scatter(coordinate[0], coordinate[1], c='b', s=5, marker='*')
    plt.title(title)
    plt.show()




