import numpy as np
import sklearn


def hi_k_means(data, b, depth):
    """
    hierarchical k means
    :param data: SIFT features from the database objects
    :param b: branch number of the vocabulary tree for each level
    :param depth: depth is the number of levels of your vocabulary tree
    :return: voc_tree: vocabulary tree
    """
    