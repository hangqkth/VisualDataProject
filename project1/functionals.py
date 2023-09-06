import cv2
import matplotlib.pyplot as plt
import numpy as np


def read_img(img_path):
    img = cv2.imread(img_path)
    return img


def show_img(img_array, img_name=None):
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
