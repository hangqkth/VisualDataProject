import numpy as np
import sklearn
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans


def k_means(data, b):
    if data.shape[0] >= b:
        kmeans = KMeans(n_clusters=b, random_state=0, max_iter=1)
        kmeans.fit(data)
        result = kmeans.labels_
        center = kmeans.cluster_centers_
        stop_flag = False
    else:
        result = []
        center = []
        stop_flag = True
    return list(result), center, stop_flag


def hi_k_means(data, obj_idx, b, depth):
    """
        hierarchical k means
        :param data: SIFT features from the database objects
        :param obj_idx: list of length = data num, indicating what object each data belong
        :param b: branch number of the vocabulary tree for each level
        :param depth: depth is the number of levels of your vocabulary tree
        :return: voc_tree: vocabulary tree

        Tree structure:
        A list containing all nodes in the tree is a representation of a tree : [{node1}, {node2}, .... ]
        A dictionary, with some attributes keys is a representation of a node :
            node1 = {'mom': index_of_mom_node, 'child': [indexes_of_child_nodes], 'own': [index_of_this_node]
             'obj_list':[list_of_obj_number_under_this_node], 'data':feature_array_under_this_node,
              'center':center_vector_of_this_node, 'leaf':bool, True if this is a leaf node}
              NOTE: if a node is a leaf, then it does not have 'feature' attribute
    """

    ##%
    TheGreatTree = []

    root_node = {'mom': 0, 'own': 0, 'obj_list': obj_idx, 'data': data, 'leaf': False}
    TheGreatTree.append(root_node)

    # start build tree
    for d in range(depth):
        # check if all nodes are leaves
        report_leaf_list = [node['leaf'] for node in TheGreatTree]
        if False not in report_leaf_list:
            print("no more dividing")
            return TheGreatTree
        else:
            for n in range(len(TheGreatTree)):
                node = TheGreatTree[n]
                if node['leaf'] is False and 'child' not in node.keys():
                    tree_idx, centers, leaf_flag = k_means(node['data'], b)
                    node['child'] = [len(TheGreatTree) + i for i in range(b)]
                    for c in range(b):  # search over three class
                        # print(c)
                        feature_idx = [i for i in range(len(tree_idx)) if tree_idx[i] == c]  # correspond object index
                        temp_data = np.concatenate(
                            [np.expand_dims(node['data'][fea, :], axis=0) for fea in feature_idx], axis=0)
                        temp_obj_idx = [node['obj_list'][fea] for fea in feature_idx]
                        center = centers[c]
                        is_leaf = False if len(feature_idx) >= b and d < depth - 1 else True
                        if is_leaf:
                            new_node = {'mom': node['own'], 'own': node['child'][c], 'obj_list': temp_obj_idx,
                                        'center': center, 'leaf': is_leaf}  # leaf node does not need to have data
                        else:
                            new_node = {'mom': node['own'], 'own': node['child'][c], 'obj_list': temp_obj_idx,
                                        'center': center, 'leaf': is_leaf, 'data': temp_data}
                        TheGreatTree.append(new_node)
                    del node['data']  # save memory
                    TheGreatTree[n] = node

    obj_list_in_classes = [node['obj_list'] for node in TheGreatTree if node['leaf']]

    return TheGreatTree, obj_list_in_classes


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
            f_ij = k_mean_obj_list[i].count(j + 1)
            w_ij = f_ij / F_j_list[j] * np.log2(obj_num / Ki)
            tf_dif_array[j, i] = w_ij
    # print(tf_dif_array)
    plt.imshow(tf_dif_array)
    plt.colorbar()
    plt.show()
    return tf_dif_array, idf_list


def search_new_tree(query_feature, old_tree, idf_vector, depth):
    # copy leaf flag from old tree, we only compare leaves later
    new_tree = [{'leaf': old_tree[n]['leaf']} for n in range(len(old_tree))]
    root_node = {'mom': 0, 'own': 0, 'data': query_feature, 'leaf': False}
    new_tree[0] = root_node
    nodes_to_search = [root_node]  # starting from root
    for d in range(depth):
        nodes_to_search_next = []
        for n in range(len(nodes_to_search)):
            node = nodes_to_search[n]
            refer_node = old_tree[node['own']]  # get the node on the same position from old tree
            child_idxes = refer_node['child']
            centers = [old_tree[c]['center'] for c in child_idxes]  # get center from children to compare distance
            cluster_idx_list = []
            for f in range(node['data'].shape[0]):  # search over all query feature
                f_distance = []
                for c in centers:  # search over current possible branches
                    d_c = np.linalg.norm(node['data'][f, ]-c, 2)
                    f_distance.append(d_c)
                cluster_idx_list.append(f_distance.index(min(f_distance)))
            for c in range(len(child_idxes)):  # classify current node feature and build new nodes
                feature_idx = [i for i in range(len(cluster_idx_list)) if cluster_idx_list[i] == c]
                if len(feature_idx) == 0:  # no match for this node, but maybe it is not a leaf
                    new_node = {'mom': node['own'], 'own': child_idxes[c], 'leaf': old_tree[child_idxes[c]]['leaf']}
                else:
                    temp_data = np.concatenate(
                        [np.expand_dims(node['data'][fea, :], axis=0) for fea in feature_idx], axis=0)
                    new_node = {'mom': node['own'], 'own': child_idxes[c], 'leaf': old_tree[child_idxes[c]]['leaf'],
                                'data': temp_data}
                    if new_node['leaf'] is False:
                        nodes_to_search_next.append(new_node)
                new_tree[child_idxes[c]] = new_node
        nodes_to_search = nodes_to_search_next
    tf_idf_vec = np.zeros_like(idf_vector)
    leaf_vec = []
    for n in range(len(new_tree)):
        current_node = new_tree[n]
        if current_node['leaf']:
            if 'data' in current_node.keys():
                leaf_vec.append(current_node['data'].shape[0])
            else:
                leaf_vec.append(0)
    for n in range(len(tf_idf_vec)):
        tf_idf_vec[n] = leaf_vec[n] / sum(leaf_vec)

    return tf_idf_vec


def match_object(weight_matrix, query_vector):
    obj_error_list = []
    for obj in range(weight_matrix.shape[0]):
        obj_error = np.linalg.norm(weight_matrix[obj] - query_vector, 1)
        obj_error_list.append(obj_error)
    predict_obj = obj_error_list.index(min(obj_error_list)) + 1
    return predict_obj


def test_tree():
    avg_recall = []
    for i in range(50):
        query_img_feature = np.load('./features/client/'+str(i+1)+'.npy')
        query_vec = search_new_tree(query_img_feature, the_tree, idf_vec, depth)
        predict_obj = match_object(tf_idf_matrix, query_vec)
        print("True object:"+str(i+1), " Predicted obj: ", predict_obj)
        avg_recall.append(1) if i+1 == predict_obj else avg_recall.append(0)
    print("average recall = ", np.average(np.array(avg_recall)))


if __name__ == "__main__":
    X = np.random.rand(1500, 128)  # 50 datapoints with feature dimension of 128
    obj_list_x = sorted((np.random.randint(10, size=1500) + 1).tolist())  # object index from 1-20 for each datapoint
    obj_feature_num_list = [obj_list_x.count(i + 1) for i in range(10)]
    b, depth = 5, 7

    # the_tree, obj_list_k_mean = hi_k_means(data=X, obj_idx=obj_list_x, b=b, depth=depth)
    # tf_idf_matrix, idf_vec = tf_idf(obj_list_k_mean, len(obj_feature_num_list), obj_feature_num_list)
    # X_query = np.random.rand(150, 128)
    # query_vec = search_new_tree(X_query, the_tree, idf_vec, depth)

    server_des_root = './features/server'
    server_feature_list = sorted(os.listdir(server_des_root))
    server_features_files = [server_des_root + '/' + str(i + 1) + '.npy' for i in range(len(server_feature_list))]
    server_features = [np.load(f) for f in server_features_files]
    server_obj_feature_num_list = [f.shape[0] for f in server_features]  # number of features in each object
    obj_list = []
    for obj in range(len(server_features)):
        for feature in range(server_features[obj].shape[0]):
            obj_list.append(obj + 1)
    server_features = np.concatenate(server_features, axis=0)

    the_tree, obj_list_k_mean = hi_k_means(data=server_features, obj_idx=obj_list, b=b, depth=depth)
    tf_idf_matrix, idf_vec = tf_idf(obj_list_k_mean, len(server_obj_feature_num_list), server_obj_feature_num_list)

    # query_img_feature = np.load('./features/client/1.npy')
    # query_vec = search_new_tree(query_img_feature, the_tree, idf_vec, depth)
    # predict_obj = match_object(tf_idf_matrix, query_vec)
    # print(predict_obj)

    test_tree()
