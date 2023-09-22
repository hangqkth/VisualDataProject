import numpy as np
import sklearn
import os
from sklearn.cluster import KMeans


def hi_k_means(data, b, depth):
    """
    hierarchical k means
    :param data: SIFT features from the database objects
    :param b: branch number of the vocabulary tree for each level
    :param depth: depth is the number of levels of your vocabulary tree
    :return: voc_tree: vocabulary tree
    """
    for d in range(depth):
        kmeans = KMeans(n_clusters=b, random_state=0, n_init="auto").fit(data)



if __name__ == "__main__":
    X = np.array([[1, 2], [1, 4], [1, 0],
                  [10, 2], [10, 4], [10, 0]])
    server_des_root = './features/server'
    server_feature_list = sorted(os.listdir(server_des_root))

    server_features_files = [server_des_root+'/'+str(i+1)+'.npy' for i in range(len(server_feature_list))]
    # print(server_features_files)
    server_features = [np.load(f) for f in server_features_files]
    # for i in server_features:
    #     print(i.shape)
    obj_list = []
    for obj in range(len(server_features)):
        for feature in range(server_features[obj].shape[0]):
            obj_list.append(obj+1)
    print(server_features[0].shape)
    print(obj_list[23100:23102])
    server_features = np.concatenate(server_features, axis=0)
    print(server_features.shape)

