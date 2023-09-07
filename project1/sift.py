import numpy as np
import cv2 as cv
from functionals import read_img, show_img, normalize
import scipy.signal
import os


data_img = read_img('./data1/obj1_5.JPG')
query_img = read_img('./data1/obj1_t1.JPG')

# show_img(data_img, 'database')
# show_img(query_img, 'query')

data_img = cv.cvtColor(data_img, cv2.COLOR_BGR2GRAY)
# sift = cv2.SIFT_create()
# kp = sift.detect(data_img, None)
# img = cv2.drawKeypoints(data_img, kp, data_img)
# print(kp[0])
surf = cv.xfeatures2d.SURF_create(400)
kp, des = surf.detectAndCompute(data_img, None)

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
        conv_out.append(normalize(scipy.signal.convolve(kernel, data_img[:, :, i], mode='same')))
    output = np.stack(conv_out, axis=-1)
    return output






