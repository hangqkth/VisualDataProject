import os
import numpy as np
import cv2


def extract_sift(img_path):
    img = cv2.imread(img_path)
    obj_num = img_path[img_path.index('j')+1:img_path.index('_')]  # 1-50
    sift = cv2.xfeatures2d.SIFT_create(edgeThreshold=3, contrastThreshold=0.08, nOctaveLayers=3)
    _, des = sift.detectAndCompute(img, None)  # des is a list of feature vector, [[128], [128], ...]
    return np.array(des), obj_num


def get_obj_feature(file_list, database):
    obj_dict = {}
    for i in range(50):
        obj_dict[str(i+1)] = []
    for f in range(len(file_list)):
        print(f)
        img_feature, obj_number = extract_sift(file_list[f])
        obj_dict[obj_number].append(img_feature)
    for i in range(50):
        obj_feature = np.concatenate(obj_dict[str(i+1)], axis=0)
        print(obj_feature.shape)
        save_path = './features/'+database+'/'+str(i+1)+'.npy'
        np.save(save_path, obj_feature)


if __name__ == "__main__":
    client_root = './Data2/client'
    server_root = './Data2/server'
    client_list = sorted(os.listdir(client_root))
    client_img_files = [os.path.join(client_root, client_list[i]) for i in range(len(client_list))]
    server_list = sorted(os.listdir(server_root))
    server_img_files = [os.path.join(server_root, server_list[i]) for i in range(len(server_list))]

    client_des_root = './features/client'
    server_des_root = './features/server'

    get_obj_feature(server_img_files, database="server")
