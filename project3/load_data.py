import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def reshape_img(vec_img):
    # the original image is of shape 32*32
    r = np.reshape(vec_img[:1024], (32, 32))
    g = np.reshape(vec_img[1024:2*1024], (32, 32))
    b = np.reshape(vec_img[2*1024:], (32, 32))
    return np.stack([r, g, b], axis=0)


def normalization(x):
    return ((x - np.min(x)) / (np.max(x) - np.min(x))) - 0.5


def load_batch_data(batch_data):
    reshape_batch_data = []
    for i in range(batch_data.shape[0]):
        reshape_batch_data.append(normalization(reshape_img(batch_data[i, ])))
    return np.array(reshape_batch_data)


class Cifar10(data.Dataset):
    def __init__(self, data_array, label_array):
        self.data = data_array
        self.label = label_array

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, item):
        return self.data[item, ], self.label[item, ]


if __name__ == "__main__":
    all_data = []
    all_label = []
    for i in range(5):
        print(i)
        test_file = './cifar-10-batches-py/data_batch_'+str(i+1)
        dict_data = unpickle(file=test_file)
        # print(dict_data[b'data'].shape)  # dict_keys([b'batch_label', b'labels', b'data', b'filenames'])
        batch_data = load_batch_data(dict_data[b'data'])
        all_data.append(batch_data)
        all_label.append(dict_data[b'labels'])
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    print(all_data.shape, all_label.shape)
    my_dataset = Cifar10(data_array=all_data, label_array=all_label)
    train_loader = data.DataLoader(dataset=my_dataset, batch_size=64, shuffle=True)
    for data, label in train_loader:
        print(data.shape)
        print(label.shape)

