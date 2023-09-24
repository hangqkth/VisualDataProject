import numpy as np
import sklearn
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans


def k_means(data, b):
    kmeans = KMeans(n_clusters=b, random_state=0, max_iter=1)
    kmeans.fit(data)
    result = kmeans.labels_
    center = kmeans.cluster_centers_
    return list(result), center


def hi_k_means(data, obj_idx, b, depth):
    """
        hierarchical k means
        :param data: SIFT features from the database objects
        :param obj_idx: list of length = data num, indicating what object each data belong
        :param b: branch number of the vocabulary tree for each level
        :param depth: depth is the number of levels of your vocabulary tree
        :return: voc_tree: vocabulary tree
    """
    temp_data_list = []
    temp_obj_list = []
    temp_center = []
    for d in range(depth):
        temp_data_list_next = []
        temp_obj_list_next = []
        temp_center_list_next = []
        if d == 0:  # first generate b branches
            tree_idx, centers = k_means(data, b)  # class list
            for c in range(b):  # search over three class
                feature_idx = [i for i in range(len(tree_idx)) if tree_idx[i] == c]  # correspond object index
                temp_data = np.concatenate([np.expand_dims(data[fea, :], axis=0) for fea in feature_idx], axis=0)
                temp_obj_idx = [obj_idx[fea] for fea in feature_idx]
                temp_data_list_next.append(temp_data)
                temp_obj_list_next.append(temp_obj_idx)
            temp_center_list_next.append(centers)
        else:
            for t in range(len(temp_data_list)):
                tree_idx, centers = k_means(temp_data_list[t], b)  # class list
                for c in range(b):  # search over three class
                    feature_idx = [i for i in range(len(tree_idx)) if tree_idx[i] == c]   # correspond object index
                    temp_data = np.concatenate([np.expand_dims(temp_data_list[t][fea, :], axis=0) for fea in feature_idx], axis=0)
                    temp_obj_idx = [temp_obj_list[t][fea] for fea in feature_idx]
                    temp_data_list_next.append(temp_data)
                    temp_obj_list_next.append(temp_obj_idx)
                temp_center_list_next.append(centers)
        temp_data_list = temp_data_list_next
        temp_obj_list = temp_obj_list_next
        temp_center.append(np.concatenate(temp_center_list_next, axis=0))  # [[b^1, feature], [b^2, feature], [b^3, feature]...]
        # print(temp_obj_list)

    return temp_obj_list, temp_center


def tf_idf(k_mean_obj_list, obj_num, obj_feature_num_list):
    words_num = len(k_mean_obj_list)
    tf_dif_array = np.zeros((obj_num, words_num))
    F_j_list = obj_feature_num_list
    idf_list = []
    for i in range(words_num):
        word_i = k_mean_obj_list[i]
        Ki = len(set(word_i))
        idf_list.append(obj_num / Ki)
        for j in range(obj_num):
            # print(j)
            f_ij = k_mean_obj_list[i].count(j+1)
            w_ij = f_ij/F_j_list[j] * np.log2(obj_num/Ki)
            tf_dif_array[j, i] = w_ij
    # print(tf_dif_array)
    plt.imshow(tf_dif_array)
    plt.colorbar()
    plt.show()
    return tf_dif_array, idf_list


def search_new_tree(query_feature, tree_center, idf_vector):
    classified_feature = []
    cluster_idx_list = []
    for depth in range(len(tree_center)):
        cluster_idx_list = []
        temp_classified_feature = [[] for i in range(tree_center[depth].shape[0])]
        if depth == 0:
            for f in range(query_feature.shape[0]):  # search over all query feature
                f_distance = []
                for c in range(tree_center[depth].shape[0]):  # search over current possible branches
                    center = tree_center[depth][c, ]
                    d_c = np.linalg.norm(query_feature[f, ]-center, 2)
                    f_distance.append(d_c)
                cluster_idx = f_distance.index(min(f_distance))
                cluster_idx_list.append(cluster_idx)  # choose the closest branch
                # classify feature according to cluster_idx
                temp_classified_feature[cluster_idx].append(query_feature[f, ])
            classified_feature = temp_classified_feature
            # print(cluster_idx_list)
        else:
            unit_branches = tree_center[0].shape[0]
            for t in range(len(classified_feature)):
                for f in range(len(classified_feature[t])):
                    f_distance = []
                    for c in range(unit_branches):
                        center = tree_center[depth][c+t*unit_branches, ]  # only search child root of current branch
                        d_c = np.linalg.norm(classified_feature[t][f] - center, 2)
                        f_distance.append(d_c)
                    cluster_idx = f_distance.index(min(f_distance))+t*unit_branches
                    cluster_idx_list.append(cluster_idx)  # choose the closest branch
                    temp_classified_feature[cluster_idx].append(classified_feature[t][f])
            classified_feature = temp_classified_feature
            # print(cluster_idx_list)
    # calculate new tf-idf vector for query object
    tf_idf_vec = np.zeros_like(idf_vector)
    for n in list(set(cluster_idx_list)):
        w_i = (cluster_idx_list.count(n)/len(cluster_idx_list))*idf_vector[n]
        tf_idf_vec[n] = w_i
    # print(tf_idf_vec)
    return tf_idf_vec


def match_object(weight_matrix, query_vector):
    obj_error_list = []
    for obj in range(weight_matrix.shape[0]):
        obj_error = np.linalg.norm(weight_matrix[obj]-query_vector, 1)
        obj_error_list.append(obj_error)
    predict_obj = obj_error_list.index(min(obj_error_list))+1
    return predict_obj




if __name__ == "__main__":
    X = np.random.rand(1500, 128)  # 50 datapoints with feature dimension of 128
    obj_list_x = sorted((np.random.randint(10, size=1500)+1).tolist())  # object index from 1-20 for each datapoint
    obj_feature_num_list = [obj_list_x.count(i+1) for i in range(10)]

    server_des_root = './features/server'
    server_feature_list = sorted(os.listdir(server_des_root))
    server_features_files = [server_des_root+'/'+str(i+1)+'.npy' for i in range(len(server_feature_list))]
    server_features = [np.load(f) for f in server_features_files]
    server_obj_feature_num_list = [f.shape[0] for f in server_features]  # number of features in each object
    obj_list = []
    for obj in range(len(server_features)):
        for feature in range(server_features[obj].shape[0]):
            obj_list.append(obj+1)
    server_features = np.concatenate(server_features, axis=0)

    obj_list_k_mean, centers = hi_k_means(data=server_features, obj_idx=obj_list, b=4, depth=4)
    # obj_list_k_mean, centers = hi_k_means(data=X, obj_idx=obj_list_x, b=4, depth=3)
    tf_idf_matrix, idf_vec = tf_idf(obj_list_k_mean, len(server_obj_feature_num_list), server_obj_feature_num_list)
    # tf_idf_matrix, idf_vec = tf_idf(obj_list_k_mean, len(obj_feature_num_list), obj_feature_num_list)
    X_query = np.random.rand(150, 128)
    query_img_feature = np.load('./features/client/1.npy')
    print(query_img_feature.shape)
    # query_vec = search_new_tree(X_query, centers, idf_vec)
    query_vec = search_new_tree(query_img_feature, centers, idf_vec)
    predict_obj = match_object(tf_idf_matrix, query_vec)
    print(predict_obj)
