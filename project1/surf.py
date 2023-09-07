import numpy as np
import cv2
from functionals import read_img, draw_kp
import scipy.signal
import os

data_img = read_img('./data1/obj1_5.JPG')
query_img = read_img('./data1/obj1_t1.JPG')

surf = cv2.xfeatures2d.SURF_create(7000)
kp_surf, des = surf.detectAndCompute(data_img, None)

kp_num = len(kp_surf)
print('key points number = ', kp_num)

draw_kp(kp_surf, data_img)