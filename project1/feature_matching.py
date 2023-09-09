import matplotlib.pyplot as plt
import numpy as np
import cv2
from functionals import read_img, draw_kp, show_img
from keypoint_robustness import sift_kp_extract, surf_kp_extract

data_img = read_img('./data1/obj1_5.JPG')
query_img = read_img('./data1/obj1_t1.JPG')

sift = cv2.xfeatures2d.SIFT_create(edgeThreshold=2.9, contrastThreshold=0.16, nOctaveLayers=5, nfeatures=400)
surf = cv2.xfeatures2d.SURF_create(hessianThreshold=8000)


def compare_keypoint():
    kp_sift_data = sift_kp_extract(data_img)
    kp_sift_query = sift_kp_extract(query_img)

    draw_kp(kp_sift_data, data_img, "obj1_5")
    draw_kp(kp_sift_query, query_img, "obj1_t5")


def fix_threshold_match():
    keypoints1, descriptors1 = sift.detectAndCompute(data_img, None)
    keypoints2, descriptors2 = sift.detectAndCompute(query_img, None)
    bf = cv2.BFMatcher()
    matches = bf.match(descriptors1, descriptors2)
    good_matches = []
    for m in matches:
        if m.distance < 200:
            good_matches.append(m)
    output_image = cv2.drawMatches(data_img, keypoints1, query_img, keypoints2, good_matches, None)
    cv2.imwrite('./matched_img/fix_threshold.jpg', output_image)


def nn_match():
    keypoints1, descriptors1 = sift.detectAndCompute(data_img, None)
    keypoints2, descriptors2 = sift.detectAndCompute(query_img, None)
    bf = cv2.BFMatcher()
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    output_image = cv2.drawMatches(data_img, keypoints1, query_img, keypoints2, matches, None)
    cv2.imwrite('./matched_img/nn_match.jpg', output_image)


def nn_ratio_match():
    keypoints1, descriptors1 = sift.detectAndCompute(data_img, None)
    keypoints2, descriptors2 = sift.detectAndCompute(query_img, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    threshold = 0.7
    good_matches = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            good_matches.append(m)
    output_image = cv2.drawMatches(data_img, keypoints1, query_img, keypoints2, good_matches, None)
    cv2.imwrite('./matched_img/nn_ratio_match.jpg', output_image)


fix_threshold_match()
nn_match()
nn_ratio_match()

