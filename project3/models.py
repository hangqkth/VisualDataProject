import torch.nn as nn
import torch
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
from load_data import Cifar10, unpickle, load_batch_data


class CifarNet(nn.Module):
    def __init__(self, c_in=3):
        super(CifarNet, self).__init__()
        self.deep_feature = 4096  # 1536
        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        self.cnn2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        self.cnn3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=self.deep_feature, out_features=512),
            nn.ReLU())
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x_in):
        x = self.cnn1(x_in)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    all_data = []
    all_label = []
    for i in range(5):
        test_file = './cifar-10-batches-py/data_batch_' + str(i + 1)
        dict_data = unpickle(file=test_file)
        # print(dict_data[b'data'].shape)  # dict_keys([b'batch_label', b'labels', b'data', b'filenames'])
        batch_data = load_batch_data(dict_data[b'data'])
        all_data.append(batch_data)
        all_label.append(dict_data[b'labels'])
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    my_dataset = Cifar10(data_array=all_data, label_array=all_label)

    train_loader = data.DataLoader(dataset=my_dataset, batch_size=64, shuffle=True)

    net = CifarNet()
    for data, label in train_loader:
        pred = net(data.to(dtype=torch.float32))
        print(pred.shape, label.shape)